import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# ---------------------------
# Aggregation Layer (unchanged)
# ---------------------------
class AggregationLayer(layers.Layer):
    """
    Aggregates a set of features over the sequence dimension.
    Supported aggregations: "mean" or "max".
    """
    def __init__(self, aggreg="mean", **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.aggreg = aggreg

    def call(self, inputs):
        if self.aggreg == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.aggreg == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError("Given aggregation string is not implemented. Use 'mean' or 'max'.")

# ---------------------------
# Attention Convolution Layer (unchanged)
# ---------------------------
class AttentionConvLayer(layers.Layer):
    """
    Applies one or more 2D convolutions on the attention scores (before softmax)
    with different filter heights. Each convolution uses a kernel whose width exactly
    matches the attention matrix’s width (proj_dim) so that the filter only moves
    along the sequence (vertical) direction.
    
    A new parameter, vertical_stride, enables you to change the stride along the vertical dimension.
    When vertical_stride > 1, the output is upsampled back to the original sequence length.
    """
    def __init__(self, filter_heights=[1], vertical_stride=1, **kwargs):
        super(AttentionConvLayer, self).__init__(**kwargs)
        self.filter_heights = filter_heights
        self.vertical_stride = vertical_stride
        self.conv_layers = []  # Will be instantiated in build()

    def build(self, input_shape):
        self.proj_dim = input_shape[-1]
        self.conv_layers = []
        for h in self.filter_heights:
            conv_layer = layers.Conv2D(
                filters=1, 
                kernel_size=(h, self.proj_dim),
                strides=(self.vertical_stride, 1),
                padding='same',
                activation=None)
            self.conv_layers.append(conv_layer)
        super(AttentionConvLayer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_heads = tf.shape(inputs)[1]
        seq_len = tf.shape(inputs)[2]
        proj_dim = tf.shape(inputs)[3]
        x = tf.reshape(inputs, (-1, seq_len, proj_dim, 1))
        
        if len(self.conv_layers) == 1:
            conv_out = self.conv_layers[0](x)
            out = tf.squeeze(conv_out, axis=-1)
        else:
            conv_outputs = [conv(x) for conv in self.conv_layers]
            stacked = tf.stack(conv_outputs, axis=-1)
            avg = tf.reduce_mean(stacked, axis=-1)
            out = tf.squeeze(avg, axis=-1)

        new_seq_len = tf.shape(out)[1]
        if self.vertical_stride > 1:
            out = tf.image.resize(tf.expand_dims(out, -1), size=(seq_len, proj_dim), method='bilinear')
            out = tf.squeeze(out, axis=-1)
        output = tf.reshape(out, (batch_size, num_heads, seq_len, proj_dim))
        return output

# ---------------------------
# Dynamic Tanh Activation (unchanged)
# ---------------------------
class DynamicTanh(layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicTanh, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha", shape=(1,), initializer='ones', trainable=True)
        self.beta = self.add_weight(name="beta", shape=(1,), initializer='zeros', trainable=True)
        super(DynamicTanh, self).build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs + self.beta)

# ---------------------------
# Clustered Linformer Attention (with share_EF support)
# ---------------------------
class ClusteredLinformerAttention(layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        proj_dim,
        cluster_E=False,
        cluster_F=False,
        share_EF=False,
        convolution=False,
        conv_filter_heights=[1,3,5],
        vertical_stride=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.proj_dim = proj_dim
        self.cluster_E = cluster_E
        self.cluster_F = cluster_F
        self.share_EF = share_EF
        self.convolution = convolution
        self.conv_filter_heights = conv_filter_heights
        self.vertical_stride = vertical_stride

    def build(self, input_shape):
        self.seq_len = input_shape[1]

        # Q/K/V weight matrices
        self.wq = self.add_weight('wq', shape=(self.d_model, self.d_model),
                                  initializer='glorot_uniform', trainable=True)
        self.wk = self.add_weight('wk', shape=(self.d_model, self.d_model),
                                  initializer='glorot_uniform', trainable=True)
        self.wv = self.add_weight('wv', shape=(self.d_model, self.d_model),
                                  initializer='glorot_uniform', trainable=True)

        # unified chunk size for both E and F
        self.chunk_size = (self.seq_len + self.proj_dim - 1) // self.proj_dim

        # E projection weights
        if not self.cluster_E:
            self.E = self.add_weight('proj_E',
                                     shape=(self.num_heads, self.seq_len, self.proj_dim),
                                     initializer='glorot_uniform', trainable=True)
        else:
            self.cluster_E_W = self.add_weight('cluster_E_W',
                                               shape=(self.num_heads, self.proj_dim, self.chunk_size),
                                               initializer='glorot_uniform', trainable=True)

        # F projection weights (or share E)
        if self.share_EF:
            if self.cluster_E:
                self.cluster_F_W = self.cluster_E_W
            else:
                self.F = self.E
        else:
            if not self.cluster_F:
                self.F = self.add_weight('proj_F',
                                         shape=(self.num_heads, self.seq_len, self.proj_dim),
                                         initializer='glorot_uniform', trainable=True)
            else:
                self.cluster_F_W = self.add_weight('cluster_F_W',
                                                   shape=(self.num_heads, self.proj_dim, self.chunk_size),
                                                   initializer='glorot_uniform', trainable=True)

        if self.convolution:
            self.attn_conv = AttentionConvLayer(self.conv_filter_heights, self.vertical_stride)

        self.dense = layers.Dense(self.d_model)

        # ⮕ CHANGED: Precompute a lookup table of indices for all possible real-token counts R=1..seq_len
        table = np.zeros((self.seq_len, self.proj_dim, self.chunk_size), dtype=np.int32)
        for R in range(1, self.seq_len + 1):
            # Assign the first R positions into proj_dim clusters evenly
            assign = np.floor(np.arange(R) * self.proj_dim / R).astype(np.int32)
            assign = np.minimum(assign, self.proj_dim - 1)
            # For each cluster, gather its indices and pad with R (we will pad k/v at position seq_len)
            for c in range(self.proj_dim):
                idxs = np.where(assign == c)[0]
                pad_len = self.chunk_size - len(idxs)
                if pad_len > 0:
                    idxs = np.concatenate([idxs, np.full(pad_len, R, dtype=np.int32)])
                table[R-1, c, :] = idxs
        self.cluster_pos_idxs = tf.constant(table, dtype=tf.int32)  # shape=(seq_len, P, chunk_size)
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x, mask=None):
        """
        x: Tensor of shape (batch, seq_len, d_model)
        mask: (optional) boolean Tensor of shape (batch, seq_len),
              where False indicates padding positions.
        """
        batch = tf.shape(x)[0]

        # 1) Linear Q/K/V
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)

        # 2) Split heads
        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        # 3) Append one zero-vector at seq_len for padding-index
        k = tf.pad(k, [[0,0],[0,0],[0,1],[0,0]])  # ⮕ CHANGED
        v = tf.pad(v, [[0,0],[0,0],[0,1],[0,0]])  # ⮕ CHANGED

        # 4) Compute number of real tokens R per sequence
        real_counts = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)      # (batch,)
        real_counts = tf.clip_by_value(real_counts, 1, self.seq_len)

        # 5) Gather the appropriate index slice for each R
        #    pos_idx: (batch, proj_dim, chunk_size)
        pos_idx = tf.gather(self.cluster_pos_idxs, real_counts - 1)      # ⮕ CHANGED

        # 6) Expand to heads and gather k/v
        #    shape => (batch, heads, proj_dim, chunk_size)
        pos_idx = tf.expand_dims(pos_idx, axis=1)                        # (batch,1,P,chunk)
        pos_idx = tf.tile(pos_idx, [1, self.num_heads, 1, 1])            # ⮕ CHANGED

        #    Now gather along seq_len axis (axis=2), preserving batch and heads
        #    Resulting shape: (batch, heads, proj_dim, chunk_size, depth)
        k_chunks = tf.gather(k, pos_idx, axis=2, batch_dims=2)           # ⮕ CHANGED
        v_chunks = tf.gather(v, pos_idx, axis=2, batch_dims=2)           # ⮕ CHANGED

        # 7) Linformer projection on each chunk
        if not self.cluster_E:
            k_proj = tf.einsum('bhnd,hnr->bhrd', k, self.E)
        else:
            k_proj = tf.einsum('bhcld,hcl->bhcd', k_chunks, self.cluster_E_W)  # ⮕ CHANGED

        if not self.cluster_F:
            v_proj = tf.einsum('bhnd,hnr->bhrd', v, self.F)
        else:
            v_proj = tf.einsum('bhcld,hcl->bhcd', v_chunks, self.cluster_F_W)  # ⮕ CHANGED

        # 8) Scaled dot-product
        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k_proj, transpose_b=True) / tf.math.sqrt(dk)
        # if mask is not None:
        #     cluster_mask0 = tf.less(
        #     pos_idx[:, :, 0],           # first element in each cluster
        #     tf.expand_dims(real_counts, 1)  # shape (batch,1)
        # )
        
        # # 2) broadcast to match `scores` shape (batch, heads, query_len, proj_dim)
        # key_mask = tf.cast(
        #     cluster_mask0[:, None, None, :],  # -> (batch,1,1,proj_dim)
        #     scores.dtype
        # )
        
        # # 3) apply
        # scores += (1.0 - key_mask) * -1e9
            
        if self.convolution:
            scores = self.attn_conv(scores)
        weights = tf.nn.softmax(scores, axis=-1)
        attn_out = tf.matmul(weights, v_proj)

        attn_out = tf.transpose(attn_out, [0,2,1,3])
        concat = tf.reshape(attn_out, (batch, -1, self.d_model))
        return self.dense(concat)

    # for debugging
    # def call(self, x, mask=None):
    #     """
    #     x: Tensor(shape=[batch, seq_len, d_model])
    #     mask: Optional bool Tensor(shape=[batch, seq_len])
    #     """
    #     # TRACE‐TIME prints (static shapes) – helpful for catching mismatches
    #     print("→ call() TRACE: x.shape =", x.shape, " mask.shape =", (mask.shape if mask is not None else None))
    
    #     batch = tf.shape(x)[0]
    
    #     # 1) Linear Q/K/V
    #     q = tf.matmul(x, self.wq)
    #     k = tf.matmul(x, self.wk)
    #     v = tf.matmul(x, self.wv)
    #     print("  after linear: q,k,v static shapes =", q.shape, k.shape, v.shape)
    
    #     # 2) Split heads
    #     q = self.split_heads(q, batch)
    #     k = self.split_heads(k, batch)
    #     v = self.split_heads(v, batch)
    #     print("  after split_heads: q,k,v static shapes =", q.shape, k.shape, v.shape)
    
    #     # 3) Pad one extra position
    #     k = tf.pad(k, [[0,0],[0,0],[0,1],[0,0]])
    #     v = tf.pad(v, [[0,0],[0,0],[0,1],[0,0]])
    #     print("  after pad: k,v static shapes =", k.shape, v.shape)
    
    #     # 4) Count real tokens
    #     real_counts = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    #     real_counts = tf.clip_by_value(real_counts, 1, self.seq_len)
    #     print("  real_counts static shape =", real_counts.shape)
    
    #     # 5) Lookup cluster index table
    #     pos_idx = tf.gather(self.cluster_pos_idxs, real_counts - 1)
    #     # pos_idx: (batch, proj_dim, chunk_size)
    #     print("  pos_idx static shape =", pos_idx.shape)
    
    #     # 6) Expand & gather per‐cluster chunks
    #     pos_idx = tf.expand_dims(pos_idx, 1)                           # (batch,1,P,chunk)
    #     pos_idx = tf.tile(pos_idx, [1, self.num_heads, 1, 1])         # (batch,heads,P,chunk)
    #     k_chunks = tf.gather(k, pos_idx, axis=2, batch_dims=2)
    #     v_chunks = tf.gather(v, pos_idx, axis=2, batch_dims=2)
    #     print("  k_chunks,v_chunks static shapes =", k_chunks.shape, v_chunks.shape)
    
    #     # 7) Linformer projections
    #     if not self.cluster_E:
    #         k_proj = tf.einsum('bhnd,hnr->bhrd', k, self.E)
    #     else:
    #         k_proj = tf.einsum('bhcld,hcl->bhcd', k_chunks, self.cluster_E_W)
    #     if not self.cluster_F:
    #         v_proj = tf.einsum('bhnd,hnr->bhrd', v, self.F)
    #     else:
    #         v_proj = tf.einsum('bhcld,hcl->bhcd', v_chunks, self.cluster_F_W)
    #     print("  k_proj,v_proj static shapes =", k_proj.shape, v_proj.shape)
    
    #     # 8) Scaled dot-product
    #     dk = tf.cast(self.depth, tf.float32)
    #     scores = tf.matmul(q, k_proj, transpose_b=True) / tf.math.sqrt(dk)
    #     print("  scores static shape =", scores.shape)
    
    #     # 9) **Correct** cluster-level mask
    #     # if mask is not None:
    #     #     # Only use the first index in each cluster to test if it's < R
    #     #     # pos_idx[:, :, 0] has shape (batch, proj_dim)
    #     #     first_idx = pos_idx[:, :, 0]
    #     #     cluster_mask0 = first_idx < tf.expand_dims(real_counts, 1)  # (batch, proj_dim)
    #     #     print("  cluster_mask0 static shape =", cluster_mask0.shape)
    #     #     key_mask = tf.cast(cluster_mask0[:, None, None, :], scores.dtype)
    #     #     print("  key_mask static shape =", key_mask.shape)
    #     #     scores += (1.0 - key_mask) * -1e9
    
    #     # 10) Optional convolution
    #     if self.convolution:
    #         scores = self.attn_conv(scores)
    
    #     # 11) Softmax & apply to v_proj
    #     weights = tf.nn.softmax(scores, axis=-1)
    #     print("  weights static shape =", weights.shape)
    #     attn_out = tf.matmul(weights, v_proj)
    #     print("  attn_out before transpose static shape =", attn_out.shape)
    
    #     # 12) Combine heads
    #     attn_out = tf.transpose(attn_out, [0, 2, 1, 3])
    #     print("  attn_out after transpose static shape =", attn_out.shape)
    #     concat = tf.reshape(attn_out, (batch, -1, self.d_model))
    #     print("  concat static shape =", concat.shape)
    
    #     return self.dense(concat)



# ---------------------------
# Transformer Block and Classifier Builder (with share_EF)
# ---------------------------
class LinformerTransformerBlock(layers.Layer):
    def __init__(self, d_model, d_ff, output_dim, num_heads, proj_dim,
                 cluster_E=False, cluster_F=False, share_EF=False,
                 convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1, **kwargs):
        super().__init__(**kwargs)
        self.attn = ClusteredLinformerAttention(
            d_model, num_heads, proj_dim,
            cluster_E, cluster_F, share_EF,
            convolution, conv_filter_heights, vertical_stride)
        self.act1 = DynamicTanh()
        self.act2 = DynamicTanh()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, x, mask=None):
        attn_out = self.attn(x, mask)
        out1 = self.act1(x + attn_out)
        ffn_out = self.ffn(out1)
        return self.act2(out1 + ffn_out)


def build_linformer_transformer_classifier(
    num_particles, feature_dim,
    d_model=16, d_ff=16, output_dim=16,
    num_heads=4, proj_dim=4,
    cluster_E=False, cluster_F=False, share_EF=False,
    convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1):
    # feature input
    inputs     = layers.Input((num_particles, feature_dim))
    # padding mask input
    mask_input = layers.Input((num_particles,), dtype=tf.bool)

    x = layers.Dense(d_model, activation='relu')(inputs)
    x = LinformerTransformerBlock(
        d_model, d_ff, output_dim, num_heads, proj_dim,
        cluster_E, cluster_F, share_EF,
        convolution, conv_filter_heights, vertical_stride
    )(x, mask=mask_input)

    x = AggregationLayer('max')(x)
    x = layers.Dense(d_model, activation='relu')(x)
    outputs = layers.Dense(5, activation='softmax')(x)

    return Model([inputs, mask_input], outputs)

