import tensorflow as tf
from tensorflow.keras import layers, Model


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


class PhysicsAttentionStructuredMesh2D(layers.Layer):
    """
    TensorFlow implementation of Physics_Attention_Structured_Mesh_2D
    for eta–phi (2D structured mesh) encoding.
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        H=101,
        W=31,
        kernel=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        inner_dim = dim_head * heads
        self.dim = dim
        self.dim_head = dim_head
        self.heads = heads
        self.slice_num = slice_num
        self.H = H
        self.W = W
        self.kernel = kernel
        self.inner_dim = inner_dim
        self.scale = dim_head ** -0.5
        self.dropout_rate = dropout

        self.softmax_last = layers.Softmax(axis=-1)
        self.dropout = layers.Dropout(dropout)

        # (1) Slice projections
        self.in_project_x = layers.Conv2D(
            filters=inner_dim,
            kernel_size=kernel,
            strides=1,
            padding="same",
            use_bias=True,
        )
        self.in_project_fx = layers.Conv2D(
            filters=inner_dim,
            kernel_size=kernel,
            strides=1,
            padding="same",
            use_bias=True,
        )
        self.in_project_slice = layers.Dense(slice_num, use_bias=True)

        # (2) Attention among slice tokens
        self.to_q = layers.Dense(dim_head, use_bias=False)
        self.to_k = layers.Dense(dim_head, use_bias=False)
        self.to_v = layers.Dense(dim_head, use_bias=False)

        # (3) Output projection
        self.to_out = tf.keras.Sequential(
            [
                layers.Dense(dim),
                layers.Dropout(dropout),
            ]
        )

    def build(self, input_shape):
        # temperature parameter: shape [1, heads, 1, 1]
        self.temperature = self.add_weight(
            name="temperature",
            shape=(1, self.heads, 1, 1),
            initializer=tf.keras.initializers.Constant(0.5),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: [B, N, C] with N == H * W
        batch_size = tf.shape(x)[0]
        num_tokens = tf.shape(x)[1]
        channels = tf.shape(x)[2]

        # Ensure N matches H * W at runtime
        tf.debugging.assert_equal(
            num_tokens, self.H * self.W, message="num_tokens must equal H * W for structured mesh"
        )

        # Reshape to image grid [B, H, W, C] then to channels-first for Conv2D if needed
        x_img = tf.reshape(x, (batch_size, self.H, self.W, channels))

        # (1) Slice: feature projections to heads
        fx_mid = self.in_project_fx(x_img)  # [B, H, W, inner_dim]
        fx_mid = tf.reshape(fx_mid, (batch_size, self.H * self.W, self.heads, self.dim_head))
        fx_mid = tf.transpose(fx_mid, perm=[0, 2, 1, 3])  # [B, Hh, N, Dh]

        x_mid = self.in_project_x(x_img)  # [B, H, W, inner_dim]
        x_mid = tf.reshape(x_mid, (batch_size, self.H * self.W, self.heads, self.dim_head))
        x_mid = tf.transpose(x_mid, perm=[0, 2, 1, 3])  # [B, Hh, N, Dh]

        # Compute slice weights over G groups
        # Apply Dense over last dim (Dh) -> [B, Hh, N, G]
        slice_logits = self.in_project_slice(x_mid)
        temp = tf.clip_by_value(self.temperature, 0.1, 5.0)
        slice_weights = self.softmax_last(slice_logits / temp)  # [B, Hh, N, G]

        # Sum of weights across tokens N: [B, Hh, G]
        slice_norm = tf.reduce_sum(slice_weights, axis=2)

        # Slice tokens: einsum over tokens dimension
        # fx_mid: [B, Hh, N, Dh], slice_weights: [B, Hh, N, G] -> [B, Hh, G, Dh]
        slice_token = tf.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (tf.expand_dims(slice_norm + 1e-5, axis=-1))

        # (2) Attention among slice tokens
        q_slice = self.to_q(slice_token)
        k_slice = self.to_k(slice_token)
        v_slice = self.to_v(slice_token)

        # Attention scores over G
        # q,k: [B, Hh, G, Dh] -> scores [B, Hh, G, G]
        scores = tf.matmul(q_slice, k_slice, transpose_b=True) * self.scale
        attn = self.softmax_last(scores)
        attn = self.dropout(attn)
        out_slice = tf.matmul(attn, v_slice)  # [B, Hh, G, Dh]

        # (3) Deslice back to tokens
        # out_slice: [B, Hh, G, Dh], slice_weights: [B, Hh, N, G] -> [B, Hh, N, Dh]
        out_x = tf.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = tf.transpose(out_x, perm=[0, 2, 1, 3])  # [B, N, Hh, Dh]
        out_x = tf.reshape(out_x, (batch_size, num_tokens, self.inner_dim))  # [B, N, Hh*Dh]

        return self.to_out(out_x)  # [B, N, dim]


def build_transolver_classifier(
    num_particles,
    feature_dim,
    output_dim=16,
    d_model=16,
    d_ff=16,
    heads=4,
    dim_head=64,
    slice_num=64,
    H=101,
    W=31,
    kernel=3,
    dropout=0.0,
    aggreg="max",
):
    """
    Simple classifier using PhysicsAttentionStructuredMesh2D for eta–phi encoding.
    Assumes num_particles == H * W.
    """
    inputs = layers.Input((num_particles, feature_dim))
    x = layers.Dense(d_model, activation="relu")(inputs)
    x = PhysicsAttentionStructuredMesh2D(
        dim=d_model,
        heads=heads,
        dim_head=dim_head,
        dropout=dropout,
        slice_num=slice_num,
        H=H,
        W=W,
        kernel=kernel,
    )(x)
    x = AggregationLayer(aggreg)(x)
    x = layers.Dense(d_model, activation="relu")(x)
    outputs = layers.Dense(output_dim, activation="sigmoid" if output_dim == 1 else "softmax")(x)
    return Model(inputs, outputs)


