import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------
# Aggregation Layer
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
# Attention Convolution Layer
# ---------------------------
import tensorflow as tf
from tensorflow.keras import layers

class AttentionConvLayer(layers.Layer):
    """
    Applies 2D convolutions on attention scores to capture local patterns along sequence.
    Parallelizes multiple filter heights with tf.vectorized_map.
    """
    def __init__(self, filter_heights=[1], vertical_stride=1, **kwargs):
        super(AttentionConvLayer, self).__init__(**kwargs)
        self.filter_heights = filter_heights
        self.vertical_stride = vertical_stride
        self.conv_layers = []

    def build(self, input_shape):
        # input_shape: (batch_size, num_heads, seq_len, proj_dim)
        _, _, self.seq_len, self.proj_dim = input_shape
        self.conv_layers = []
        for h in self.filter_heights:
            conv = layers.Conv2D(
                filters=1,
                kernel_size=(h, self.proj_dim),
                strides=(self.vertical_stride, 1),
                padding='same',
                activation=None
            )
            self.conv_layers.append(conv)
        super(AttentionConvLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs: (B, H, N, P)
        shape = tf.shape(inputs)
        batch, heads, seq_len, proj_dim = shape[0], shape[1], shape[2], shape[3]
        # reshape for Conv2D: (B*H, N, P, 1)
        x = tf.reshape(inputs, (-1, seq_len, proj_dim, 1))

        n_filters = len(self.conv_layers)
        indices = tf.range(n_filters, dtype=tf.int32)

        # vectorized mapping of each conv layer over the list of filters
        @tf.function
        def apply_conv(i):
            return self.conv_layers[i](x)  # (B*H, N', P, 1)

        # outs: (n_filters, B*H, N', P, 1)
        outs = tf.vectorized_map(apply_conv, indices)
        # average across filter dimension
        avg = tf.reduce_mean(outs, axis=0)      # -> (B*H, N', P, 1)
        out = tf.squeeze(avg, -1)               # -> (B*H, N', P)

        # optional resize if vertical_stride != 1
        def resize():
            resized = tf.image.resize(
                tf.expand_dims(out, -1),
                size=(seq_len, proj_dim),
                method='bilinear'
            )
            return tf.squeeze(resized, -1)

        if self.vertical_stride > 1:
            new_seq = tf.shape(out)[1]
            out = tf.cond(
                tf.equal(new_seq, seq_len),
                lambda: out,
                resize
            )

        # reshape back to (B, H, N, P)
        return tf.reshape(out, (batch, heads, seq_len, proj_dim))



# ---------------------------
# Dynamic Tanh Activation
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
# Clustered Linformer Attention
# ---------------------------
class ClusteredLinformerAttention(layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        proj_dim,
        cluster_E=False,
        cluster_F=False,
        convolution=False,
        conv_filter_heights=[1],
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
        self.convolution = convolution
        self.conv_filter_heights = conv_filter_heights
        self.vertical_stride = vertical_stride

    def build(self, input_shape):
        self.seq_len = input_shape[1]
        # Q, K, V
        self.wq = self.add_weight(shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True, name='wq')
        self.wk = self.add_weight(shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True, name='wk')
        self.wv = self.add_weight(shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True, name='wv')

        # E projection
        if not self.cluster_E:
            self.E = self.add_weight(shape=(self.num_heads, self.seq_len, self.proj_dim), initializer='glorot_uniform', trainable=True, name='proj_E')
        else:
            self.chunk_size_E = (self.seq_len + self.proj_dim - 1) // self.proj_dim
            self.cluster_E_W = self.add_weight(shape=(self.num_heads, self.proj_dim, self.chunk_size_E), initializer='glorot_uniform', trainable=True, name='cluster_E_W')

        # F projection
        if not self.cluster_F:
            self.F = self.add_weight(shape=(self.num_heads, self.seq_len, self.proj_dim), initializer='glorot_uniform', trainable=True, name='proj_F')
        else:
            self.chunk_size_F = (self.seq_len + self.proj_dim - 1) // self.proj_dim
            self.cluster_F_W = self.add_weight(shape=(self.num_heads, self.proj_dim, self.chunk_size_F), initializer='glorot_uniform', trainable=True, name='cluster_F_W')

        # Optional convolution on attention scores
        if self.convolution:
            self.attn_conv = AttentionConvLayer(filter_heights=self.conv_filter_heights, vertical_stride=self.vertical_stride)

        self.dense = layers.Dense(self.d_model)
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch = tf.shape(x)[0]
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)
        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        # Project keys
        if not self.cluster_E:
            k_proj = tf.einsum('bhnd,hnr->bhrd', k, self.E)
        else:
            pad_k = self.chunk_size_E * self.proj_dim - self.seq_len
            k_p = tf.pad(k, [[0,0],[0,0],[0,pad_k],[0,0]])
            k_chunks = tf.reshape(k_p, (batch, self.num_heads, self.proj_dim, self.chunk_size_E, self.depth))
            k_proj = tf.einsum('bhcld,hcl->bhcd', k_chunks, self.cluster_E_W)

        # Project values
        if not self.cluster_F:
            v_proj = tf.einsum('bhnd,hnr->bhrd', v, self.F)
        else:
            pad_v = self.chunk_size_F * self.proj_dim - self.seq_len
            v_p = tf.pad(v, [[0,0],[0,0],[0,pad_v],[0,0]])
            v_chunks = tf.reshape(v_p, (batch, self.num_heads, self.proj_dim, self.chunk_size_F, self.depth))
            v_proj = tf.einsum('bhcld,hcl->bhcd', v_chunks, self.cluster_F_W)

        # Attention scores
        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k_proj, transpose_b=True) / tf.math.sqrt(dk)
        if self.convolution:
            scores = self.attn_conv(scores)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_out = tf.matmul(attn_weights, v_proj)

        # Merge heads
        attn_out = tf.transpose(attn_out, perm=[0,2,1,3])
        concat = tf.reshape(attn_out, (batch, -1, self.d_model))
        return self.dense(concat)


# ---------------------------
# Linformer Transformer Block
# ---------------------------
class LinformerTransformerBlock(layers.Layer):
    def __init__(self, d_model, d_ff, output_dim, num_heads, proj_dim,
                 cluster_E=False, cluster_F=False,
                 convolution=False, conv_filter_heights=[1], vertical_stride=1,
                 **kwargs):
        super(LinformerTransformerBlock, self).__init__(**kwargs)
        self.attention = ClusteredLinformerAttention(
            d_model, num_heads, proj_dim,
            cluster_E, cluster_F,
            convolution, conv_filter_heights, vertical_stride
        )
        self.dTanh1 = DynamicTanh()
        self.dTanh2 = DynamicTanh()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, x):
        attn_out = self.attention(x)
        out1 = self.dTanh1(x + attn_out)
        ffn_out = self.ffn(out1)
        out2 = self.dTanh2(out1 + ffn_out)
        return out2


# ---------------------------
# Build Linformer Classifier
# ---------------------------
def build_linformer_transformer_classifier(
    num_particles,
    feature_dim,
    d_model=16,
    d_ff=16,
    output_dim=16,
    num_heads=8,
    proj_dim=8,
    cluster_E=False,
    cluster_F=False,
    convolution=False,
    conv_filter_heights=[1,3,5,7],
    vertical_stride=1
):
    inputs = layers.Input(shape=(num_particles, feature_dim))
    x = layers.Dense(d_model, activation='relu')(inputs)
    x = LinformerTransformerBlock(
        d_model, d_ff, output_dim,
        num_heads, proj_dim,
        cluster_E, cluster_F,
        convolution, conv_filter_heights)(x)
    pooled = AggregationLayer(aggreg='max')(x)
    x = layers.Dense(d_model, activation='relu')(pooled)
    outputs = layers.Dense(5, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)
