import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------
# Aggregation Layer
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
# Attention Convolution Layer
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
# Dynamic Tanh Activation
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
# Multi-Head Attention with Optional Pairwise Mask
# ---------------------------
class StandardMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, convolution=False, conv_filter_heights=[1, 3, 5],
                 vertical_stride=1, use_attention_mask=True, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.d_model = d_model
        self.convolution = convolution
        self.conv_filter_heights = conv_filter_heights
        self.vertical_stride = vertical_stride
        self.use_attention_mask = use_attention_mask

        self.wq = self.add_weight("wq", shape=(d_model, d_model), initializer="glorot_uniform", trainable=True)
        self.wk = self.add_weight("wk", shape=(d_model, d_model), initializer="glorot_uniform", trainable=True)
        self.wv = self.add_weight("wv", shape=(d_model, d_model), initializer="glorot_uniform", trainable=True)
        self.dense = layers.Dense(d_model)

        if self.convolution:
            self.attn_conv = AttentionConvLayer(self.conv_filter_heights, self.vertical_stride)

        if self.use_attention_mask:
            self.mask_conv1 = layers.Conv2D(8, (1, 1), activation='relu', padding='same')
            self.mask_conv2 = layers.Conv2D(8, (1, 1), activation='relu', padding='same')
            self.mask_conv3 = layers.Conv2D(1, (1, 1), padding='same')  # FIXED: Output channel = 1

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def compute_pairwise_features(self, particles):
        pt = particles[..., 0]
        eta = particles[..., 1]
        phi = particles[..., 2]

        pt_i, pt_j = tf.expand_dims(pt, 2), tf.expand_dims(pt, 1)
        eta_i, eta_j = tf.expand_dims(eta, 2), tf.expand_dims(eta, 1)
        phi_i, phi_j = tf.expand_dims(phi, 2), tf.expand_dims(phi, 1)

        d_eta = eta_i - eta_j
        d_phi = tf.math.atan2(tf.sin(phi_i - phi_j), tf.cos(phi_i - phi_j))
        delta = tf.sqrt(d_eta ** 2 + d_phi ** 2)
        kT = tf.minimum(pt_i, pt_j) * delta
        z = tf.minimum(pt_i, pt_j) / (pt_i + pt_j + 1e-8)

        m2 = pt_i ** 2 + pt_j ** 2 - 2 * pt_i * pt_j * tf.cos(d_phi)
        m2 = tf.maximum(m2, 1e-8)  # FIXED: prevent negative or zero for log

        return tf.stack([
            tf.math.log(tf.maximum(delta, 1e-8)),
            tf.math.log(tf.maximum(kT, 1e-8)),
            tf.math.log(tf.maximum(z, 1e-8)),
            tf.math.log(m2)
        ], axis=-1)

    def call(self, x, particles=None):
        batch_size = tf.shape(x)[0]
        q = tf.matmul(x, self.wq)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)

        if self.convolution:
            scores = self.attn_conv(scores)

        if self.use_attention_mask and particles is not None:
            features = self.compute_pairwise_features(particles)  # shape: [B, S, S, 4]
            mask = self.mask_conv1(features)
            mask = self.mask_conv2(mask)
            mask = self.mask_conv3(mask)  # shape: [B, S, S, 1]
            mask = tf.squeeze(mask, axis=-1)  # shape: [B, S, S]
            mask = tf.expand_dims(mask, axis=1)  # shape: [B, 1, S, S]
            mask = tf.clip_by_value(mask, -10.0, 10.0)  # FIXED: prevent softmax explosion
            scores += mask

        weights = tf.nn.softmax(scores, axis=-1)
        attn_output = tf.matmul(weights, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat = tf.reshape(attn_output, (batch_size, -1, self.d_model))
        return self.dense(concat)

# ---------------------------
# Transformer Block
# ---------------------------
class StandardTransformerBlock(layers.Layer):
    def __init__(self, d_model, d_ff, output_dim, num_heads,
                 convolution=False, conv_filter_heights=[1, 3, 5], vertical_stride=1,
                 use_attention_mask=True, **kwargs):
        super().__init__(**kwargs)
        self.attn = StandardMultiHeadAttention(
            d_model, num_heads,
            convolution=convolution,
            conv_filter_heights=conv_filter_heights,
            vertical_stride=vertical_stride,
            use_attention_mask=use_attention_mask
        )
        self.act1 = DynamicTanh()
        self.act2 = DynamicTanh()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, x, particles=None):
        attn_out = self.attn(x, particles)
        out1 = self.act1(x + attn_out)
        ffn_out = self.ffn(out1)
        return self.act2(out1 + ffn_out)

# ---------------------------
# Model Builder
# ---------------------------
def build_standard_transformer_classifier(
    num_particles, feature_dim,
    d_model=16, d_ff=16, output_dim=16,
    num_heads=4,
    convolution=False, conv_filter_heights=[1, 3, 5], vertical_stride=1,
    use_attention_mask=True):

    inputs = layers.Input((num_particles, feature_dim))
    x = layers.Dense(d_model, activation='relu')(inputs)
    x = StandardTransformerBlock(
        d_model, d_ff, output_dim, num_heads,
        convolution=convolution,
        conv_filter_heights=conv_filter_heights,
        vertical_stride=vertical_stride,
        use_attention_mask=use_attention_mask
    )(x, particles=inputs)
    x = AggregationLayer('max')(x)
    x = layers.Dense(d_model, activation='relu')(x)
    outputs = layers.Dense(output_dim, activation='sigmoid' if output_dim == 1 else 'softmax')(x)
    return Model(inputs, outputs)