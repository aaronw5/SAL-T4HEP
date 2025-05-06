import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# ---------------------------
# Aggregation Layer
# ---------------------------
class AggregationLayer(layers.Layer):
    """
    Aggregates a set of features over the sequence dimension.
    Supported aggregations: "mean" or "max".
    """
    def __init__(self, aggreg="mean", **kwargs):
        super().__init__(**kwargs)
        self.aggreg = aggreg

    def call(self, inputs):
        if self.aggreg == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.aggreg == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError("Given aggregation string not implemented. Use 'mean' or 'max'.")

# ---------------------------
# Attention Convolution Layer
# ---------------------------
class AttentionConvLayer(layers.Layer):
    """
    Applies 2D convolutions on attention scores with kernels spanning the projection dimension.
    """
    def __init__(self, filter_heights=[1], vertical_stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filter_heights = filter_heights
        self.vertical_stride = vertical_stride
        self.conv_layers = []

    def build(self, input_shape):
        seq_len, proj_dim = input_shape[-2], input_shape[-1]
        for h in self.filter_heights:
            conv = layers.Conv2D(
                filters=1,
                kernel_size=(h, proj_dim),
                strides=(self.vertical_stride, 1),
                padding='same',
                activation=None
            )
            self.conv_layers.append(conv)
        super().build(input_shape)

    def call(self, inputs):
        batch, heads, seq_len, proj_dim = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        x = tf.reshape(inputs, (batch * heads, seq_len, proj_dim, 1))
        if len(self.conv_layers) == 1:
            out = tf.squeeze(self.conv_layers[0](x), axis=-1)
        else:
            outs = [conv(x) for conv in self.conv_layers]
            stacked = tf.stack(outs, axis=-1)
            out = tf.squeeze(tf.reduce_mean(stacked, axis=-1), axis=-1)
        if self.vertical_stride > 1:
            out = tf.image.resize(tf.expand_dims(out, -1), size=(seq_len, proj_dim), method='bilinear')
            out = tf.squeeze(out, axis=-1)
        return tf.reshape(out, (batch, heads, seq_len, proj_dim))

# ---------------------------
# Dynamic Tanh Activation
# ---------------------------
class DynamicTanh(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha", shape=(1,), initializer='ones', trainable=True)
        self.beta  = self.add_weight(name="beta",  shape=(1,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs + self.beta)

# ---------------------------
# Clustered Linformer Attention (with partition table)
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
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.depth      = d_model // num_heads
        self.proj_dim   = proj_dim
        self.cluster_E  = cluster_E
        self.cluster_F  = cluster_F
        self.share_EF   = share_EF
        if self.share_EF:
            self.cluster_F = self.cluster_E
        self.convolution         = convolution
        self.conv_filter_heights = conv_filter_heights
        self.vertical_stride     = vertical_stride

    def build(self, input_shape):
        self.seq_len = input_shape[1]
        # Q/K/V weight matrices
        self.wq = self.add_weight('wq', shape=(self.d_model, self.d_model), initializer='glorot_uniform')
        self.wk = self.add_weight('wk', shape=(self.d_model, self.d_model), initializer='glorot_uniform')
        self.wv = self.add_weight('wv', shape=(self.d_model, self.d_model), initializer='glorot_uniform')
        # projection chunk size
        self.chunk_size = (self.seq_len + self.proj_dim - 1) // self.proj_dim
        # E projection
        if self.cluster_E:
            self.cluster_E_W = self.add_weight(
                'cluster_E_W', shape=(self.num_heads, self.proj_dim, self.chunk_size), initializer='glorot_uniform'
            )
        else:
            self.E = self.add_weight(
                'proj_E', shape=(self.num_heads, self.seq_len, self.proj_dim), initializer='glorot_uniform'
            )
        # F projection or share
        if self.share_EF:
            if self.cluster_E:
                self.cluster_F_W = self.cluster_E_W
            else:
                self.F = self.E
        else:
            if self.cluster_F:
                self.cluster_F_W = self.add_weight(
                    'cluster_F_W', shape=(self.num_heads, self.proj_dim, self.chunk_size), initializer='glorot_uniform'
                )
            else:
                self.F = self.add_weight(
                    'proj_F', shape=(self.num_heads, self.seq_len, self.proj_dim), initializer='glorot_uniform'
                )
        # convolution on scores
        if self.convolution:
            self.attn_conv = AttentionConvLayer(
                filter_heights=self.conv_filter_heights,
                vertical_stride=self.vertical_stride
            )
        self.dense = layers.Dense(self.d_model)
        # precompute cluster positions table
        table = np.zeros((self.seq_len, self.proj_dim, self.chunk_size), dtype=np.int32)
        for R in range(1, self.seq_len + 1):
            assign = np.floor(np.arange(R) * self.proj_dim / R).astype(np.int32)
            assign = np.minimum(assign, self.proj_dim - 1)
            for c in range(self.proj_dim):
                idxs = np.where(assign == c)[0]
                pad_len = self.chunk_size - len(idxs)
                if pad_len > 0:
                    idxs = np.concatenate([idxs, np.full(pad_len, R, dtype=np.int32)])
                table[R-1, c, :] = idxs
        self.cluster_pos_idxs = tf.constant(table, dtype=tf.int32)
        super().build(input_shape)

    def split_heads(self, x):
        batch = tf.shape(x)[0]
        return tf.transpose(
            tf.reshape(x, (batch, -1, self.num_heads, self.depth)), [0,2,1,3]
        )

    def call(self, x, mask=None):
        batch = tf.shape(x)[0]
        # linear projections
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)
        # split heads
        q = self.split_heads(q)
        k0 = self.split_heads(k)
        v0 = self.split_heads(v)
        # pad for cluster index R
        k = tf.pad(k0, [[0,0],[0,0],[0,1],[0,0]])
        v = tf.pad(v0, [[0,0],[0,0],[0,1],[0,0]])
        # count real tokens
        if mask is None:
            real_counts = tf.fill([batch], self.seq_len)
        else:
            real_counts = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
            real_counts = tf.clip_by_value(real_counts, 1, self.seq_len)
        # gather cluster positions
        pos_idx = tf.gather(self.cluster_pos_idxs, real_counts - 1)
        pos_idx = tf.expand_dims(pos_idx, 1)
        pos_idx = tf.tile(pos_idx, [1, self.num_heads, 1, 1])
        k_chunks = tf.gather(k, pos_idx, axis=2, batch_dims=2)
        v_chunks = tf.gather(v, pos_idx, axis=2, batch_dims=2)
        # projections E/F
        if self.cluster_E:
            k_proj = tf.einsum('bhcld,hcl->bhcd', k_chunks, self.cluster_E_W)
        else:
            k_proj = tf.einsum('bhnd,hnr->bhrd', k0, self.E)
        if self.cluster_F:
            v_proj = tf.einsum('bhcld,hcl->bhcd', v_chunks, self.cluster_F_W)
        else:
            v_proj = tf.einsum('bhnd,hnr->bhrd', v0, self.F)
        # attention scores and apply conv
        dk     = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k_proj, transpose_b=True) / tf.math.sqrt(dk)
        if self.convolution:
            scores = self.attn_conv(scores)
        weights  = tf.nn.softmax(scores, axis=-1)
        attn_out = tf.matmul(weights, v_proj)
        # combine heads
        attn_out = tf.transpose(attn_out, [0,2,1,3])
        concat   = tf.reshape(attn_out, (batch, -1, self.d_model))
        return self.dense(concat)

# ---------------------------
# Transformer Block
# ---------------------------
class LinformerTransformerBlock(layers.Layer):
    def __init__(self, d_model, d_ff, output_dim, num_heads, proj_dim,
                 cluster_E=False, cluster_F=False, share_EF=False,
                 convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1, **kwargs):
        super().__init__(**kwargs)
        self.attn = ClusteredLinformerAttention(
            d_model, num_heads, proj_dim,
            cluster_E, cluster_F, share_EF,
            convolution, conv_filter_heights, vertical_stride
        )
        self.act1 = DynamicTanh()
        self.act2 = DynamicTanh()
        self.ffn  = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, x, mask=None):
        attn_out = self.attn(x, mask)
        out1     = self.act1(x + attn_out)
        ffn_out  = self.ffn(out1)
        return self.act2(out1 + ffn_out)

# ---------------------------
# Build classifier
# ---------------------------
def build_linformer_transformer_classifier(
    num_particles, feature_dim,
    d_model=16, d_ff=16, output_dim=16,
    num_heads=4, proj_dim=4,
    cluster_E=False, cluster_F=False, share_EF=False,
    convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1):
    inputs     = layers.Input((num_particles, feature_dim))
    mask_input = layers.Input((num_particles,), dtype=tf.bool)

    x = layers.Dense(d_model, activation='relu')(inputs)
    x = LinformerTransformerBlock(
        d_model, d_ff, output_dim,
        num_heads, proj_dim,
        cluster_E, cluster_F, share_EF,
        convolution, conv_filter_heights, vertical_stride
    )(x, mask=mask_input)

    x = AggregationLayer('max')(x)
    x = layers.Dense(d_model, activation='relu')(x)

    # final classifier head: uses output_dim
    activation = 'sigmoid' if output_dim == 1 else 'softmax'
    outputs = layers.Dense(output_dim, activation=activation)(x)

    return Model([inputs, mask_input], outputs)
