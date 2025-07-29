import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------
# Aggregation Layer (unchanged)
# ---------------------------
class AggregationLayer(layers.Layer):
    def __init__(self, aggreg="mean", **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.aggreg = aggreg

    def call(self, inputs):
        if self.aggreg == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.aggreg == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError("Unsupported aggregation: use 'mean' or 'max'.")

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
        for h in self.filter_heights:
            self.conv_layers.append(
                layers.Conv2D(
                    filters=1,
                    kernel_size=(h, self.proj_dim),
                    strides=(self.vertical_stride, 1),
                    padding='same',
                    activation=None
                )
            )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        num_heads = tf.shape(inputs)[1]
        seq_len = tf.shape(inputs)[2]
        proj_dim = tf.shape(inputs)[3]
        x = tf.reshape(inputs, (-1, seq_len, proj_dim, 1))

        if len(self.conv_layers) == 1:
            out = tf.squeeze(self.conv_layers[0](x), axis=-1)
        else:
            conv_outputs = [conv(x) for conv in self.conv_layers]
            stacked = tf.stack(conv_outputs, axis=-1)
            avg = tf.reduce_mean(stacked, axis=-1)
            out = tf.squeeze(avg, axis=-1)

        if self.vertical_stride > 1:
            out = tf.image.resize(tf.expand_dims(out, -1), size=(seq_len, proj_dim), method='bilinear')
            out = tf.squeeze(out, axis=-1)
        return tf.reshape(out, (batch_size, num_heads, seq_len, proj_dim))

# ---------------------------
# Dynamic Tanh Activation (unchanged)
# ---------------------------
class DynamicTanh(layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicTanh, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha", shape=(1,), initializer='ones', trainable=True)
        self.beta = self.add_weight(name="beta", shape=(1,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs + self.beta)

# ---------------------------
# Standard Transformer Attention Layer
# ---------------------------
class StandardMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model
        self.convolution = convolution
        self.conv_filter_heights = conv_filter_heights
        self.vertical_stride = vertical_stride

    def build(self, input_shape):
        self.wq = self.add_weight("wq", shape=(self.d_model, self.d_model), initializer="glorot_uniform", trainable=True)
        self.wk = self.add_weight("wk", shape=(self.d_model, self.d_model), initializer="glorot_uniform", trainable=True)
        self.wv = self.add_weight("wv", shape=(self.d_model, self.d_model), initializer="glorot_uniform", trainable=True)
        self.dense = layers.Dense(self.d_model)

        if self.convolution:
            self.attn_conv = AttentionConvLayer(self.conv_filter_heights, self.vertical_stride)

        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]

        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)

        weights = tf.nn.softmax(scores, axis=-1)
        
        if self.convolution:
            scores = self.attn_conv(scores)
            
        attn_output = tf.matmul(weights, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat = tf.reshape(attn_output, (batch_size, -1, self.d_model))
        return self.dense(concat)

# ---------------------------
# Transformer Block and Classifier Builder
# ---------------------------
class StandardTransformerBlock(layers.Layer):
    def __init__(self, d_model, d_ff, output_dim, num_heads, convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1, **kwargs):
        super().__init__(**kwargs)
        self.attn = StandardMultiHeadAttention(
            d_model, num_heads,
            convolution=convolution,
            conv_filter_heights=conv_filter_heights,
            vertical_stride=vertical_stride
        )
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

def build_standard_transformer_classifier(
    num_particles, feature_dim,
    d_model=16, d_ff=16, output_dim=16,
    num_heads=4,
    convolution=False, conv_filter_heights=[1,3,5], vertical_stride=1):
    
    inputs = layers.Input((num_particles, feature_dim))
    x = layers.Dense(d_model, activation='relu')(inputs)
    x = StandardTransformerBlock(
        d_model, d_ff, output_dim, num_heads,
        convolution=convolution,
        conv_filter_heights=conv_filter_heights,
        vertical_stride=vertical_stride
    )(x)
    x = AggregationLayer('max')(x)
    x = layers.Dense(d_model, activation='relu')(x)
    if output_dim == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    outputs = layers.Dense(output_dim, activation=activation)(x)
    return Model(inputs, outputs)
