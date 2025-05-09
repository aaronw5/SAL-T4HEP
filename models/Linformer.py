import tensorflow as tf
from tensorflow.keras import layers, Model

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

    def __init__(self, filter_heights=[1], vertical_stride=1, **kwargs):
        super(AttentionConvLayer, self).__init__(**kwargs)
        self.filter_heights = filter_heights
        self.vertical_stride = vertical_stride
        self.conv_layers = []  

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
        # Q,K,V weights
        self.wq = self.add_weight(name='wq', shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True)
        self.wk = self.add_weight(name='wk', shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True)
        self.wv = self.add_weight(name='wv', shape=(self.d_model, self.d_model), initializer='glorot_uniform', trainable=True)
        self.chunk_size_F = (self.seq_len + self.proj_dim - 1) // self.proj_dim
        self.chunk_size_E = (self.seq_len + self.proj_dim - 1) // self.proj_dim

        # E and F projections (optionally shared)
        if not self.cluster_E:
            # create E
            self.E = self.add_weight(name='proj_E', shape=(self.num_heads, self.seq_len, self.proj_dim), initializer='glorot_uniform', trainable=True)
        else:
            self.cluster_E_W = self.add_weight(name='cluster_E_W', shape=(self.num_heads, self.proj_dim, self.chunk_size_E), initializer='glorot_uniform', trainable=True)

        if self.share_EF:
            # share E and F
            if self.cluster_E:
                self.cluster_F_W = self.cluster_E_W
            else:
                self.F = self.E
        else:
            if not self.cluster_F:
                self.F = self.add_weight(name='proj_F', shape=(self.num_heads, self.seq_len, self.proj_dim), initializer='glorot_uniform', trainable=True)
            else:
                self.cluster_F_W = self.add_weight(name='cluster_F_W', shape=(self.num_heads, self.proj_dim, self.chunk_size_F), initializer='glorot_uniform', trainable=True)

        if self.convolution:
            self.attn_conv = AttentionConvLayer(self.conv_filter_heights, self.vertical_stride)

        self.dense = layers.Dense(self.d_model)
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, [0, 2, 1, 3])

    def call(self, x):
        batch = tf.shape(x)[0]
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)
        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        # key projection
        if not self.cluster_E:
            k_proj = tf.einsum('bhnd,hnr->bhrd', k, self.E)
        else:
            pad_k = self.chunk_size_E * self.proj_dim - self.seq_len
            k_p = tf.pad(k, [[0,0],[0,0],[0,pad_k],[0,0]])
            k_chunks = tf.reshape(k_p, (batch, self.num_heads, self.proj_dim, self.chunk_size_E, self.depth))
            k_proj = tf.einsum('bhcld,hcl->bhcd', k_chunks, self.cluster_E_W)

        # value projection
        if not self.cluster_F:
            v_proj = tf.einsum('bhnd,hnr->bhrd', v, self.F)
        else:
            pad_v = self.chunk_size_F * self.proj_dim - self.seq_len
            v_p = tf.pad(v, [[0,0],[0,0],[0,pad_v],[0,0]])
            v_chunks = tf.reshape(v_p, (batch, self.num_heads, self.proj_dim, self.chunk_size_F, self.depth))
            v_proj = tf.einsum('bhcld,hcl->bhcd', v_chunks, self.cluster_F_W)

        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k_proj, transpose_b=True) / tf.math.sqrt(dk)
        if self.convolution:
            scores = self.attn_conv(scores)
        weights = tf.nn.softmax(scores, axis=-1)
        attn_out = tf.matmul(weights, v_proj)

        attn_out = tf.transpose(attn_out, [0,2,1,3])
        concat = tf.reshape(attn_out, (batch, -1, self.d_model))
        return self.dense(concat)

# ---------------------------
# Transformer Block and Classifier Builder
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

    def call(self, x):
        attn_out = self.attn(x)
        out1 = self.act1(x + attn_out)
        ffn_out = self.ffn(out1)
        return self.act2(out1 + ffn_out)


def build_linformer_transformer_classifier(
    num_particles, feature_dim,
    d_model=16, d_ff=16, output_dim=5,
    num_heads=4, proj_dim=4,
    cluster_E=False, cluster_F=False, share_EF=False,
    convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1):
    inputs = layers.Input((num_particles, feature_dim))
    x = layers.Dense(d_model, activation='relu')(inputs)
    x = LinformerTransformerBlock(
        d_model, d_ff, output_dim, num_heads, proj_dim,
        cluster_E, cluster_F, share_EF,
        convolution, conv_filter_heights, vertical_stride)(x)
    x = AggregationLayer('max')(x)
    x = layers.Dense(d_model, activation='relu')(x)
    activation = ''
    if output_dim == 1: 
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    outputs = layers.Dense(output_dim, activation= activation)(x)
    return Model(inputs, outputs)
