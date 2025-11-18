import tensorflow as tf


class AggregationLayer(tf.keras.layers.Layer):
    def __init__(self, aggreg="mean", **kwargs):
        super().__init__(**kwargs)
        self.aggreg = aggreg

    def call(self, inputs):
        if self.aggreg == "mean":
            return tf.reduce_mean(inputs, axis=1)
        if self.aggreg == "max":
            return tf.reduce_max(inputs, axis=1)
        raise ValueError("Aggregation must be 'mean' or 'max'.")


class DynamicTanh(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=(1,), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return tf.math.tanh(self.alpha * inputs + self.beta)


class DeltaNetAttention(tf.keras.layers.Layer):
    """
    DeltaNet attention layer with chunkwise parallel training in TensorFlow.

    Implements UT transform per Yang et al. 2024 (Parallel DeltaNet):
    T = (I + tril(K_beta K^T, -1))^{-1} diag(beta)
    W = T K_beta, U = T V_beta

    Chunk update across sequence:
      - intra-chunk using causal scores Q K^T applied to (U - W S)
      - inter-chunk via Q S, with S accumulated as S += K^T (U - W S)
    """

    def __init__(self, d_model, num_heads, chunk_size=64, conv_kernel=None, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.chunk_size = chunk_size
        self.conv_kernel = conv_kernel

        # Projections
        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False)
        self.w_beta = tf.keras.layers.Dense(num_heads, use_bias=False)

        # Optional lightweight depthwise conv (applied per token)
        if conv_kernel and conv_kernel > 1:
            self.conv_q = tf.keras.layers.DepthwiseConv1D(
                kernel_size=conv_kernel, padding="same"
            )
            self.conv_k = tf.keras.layers.DepthwiseConv1D(
                kernel_size=conv_kernel, padding="same"
            )
            self.conv_v = tf.keras.layers.DepthwiseConv1D(
                kernel_size=conv_kernel, padding="same"
            )
        else:
            self.conv_q = None
            self.conv_k = None
            self.conv_v = None

        self.wo = tf.keras.layers.Dense(d_model, use_bias=False)

    @staticmethod
    def _feature_map_with_norm(x):
        x = tf.nn.silu(x)
        norm = tf.norm(x, ord=2, axis=-1, keepdims=True)
        return x / (norm + 1e-8)

    def _reshape_heads(self, x):
        # [B, T, D] -> [B, H, T, Dh]
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]
        x = tf.reshape(x, [b, t, self.num_heads, self.d_head])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x):
        # [B, H, T, Dh] -> [B, T, D]
        b = tf.shape(x)[0]
        t = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, t, self.d_model])

    def _compute_ut_transform(self, K, V, beta):
        # K,V: [B,H,C,Dh], beta: [B,H,C]
        K_beta = K * beta[..., None]
        V_beta = V * beta[..., None]

        # KK^T lower strictly triangular
        KK_T = tf.einsum("bhcd,bhce->bhde", K_beta, K)  # [B,H,C,C]
        L = tf.linalg.band_part(KK_T, -1, 0) - tf.linalg.band_part(KK_T, 0, 0)

        # Solve (I + L) X = I via triangular solve
        # Use static chunk_size instead of dynamic tf.shape to avoid autograph issues
        b = tf.shape(K)[0]
        h = tf.shape(K)[1]
        c = self.chunk_size  # Use static chunk_size
        eye = tf.eye(c, dtype=K.dtype)
        eye = tf.reshape(eye, [1, 1, c, c])
        eye = tf.tile(eye, [b, h, 1, 1])
        I_plus_L = eye + L
        T_left = tf.linalg.triangular_solve(
            I_plus_L, eye, lower=True
        )  # [B,H,C,C]
        beta_diag = tf.linalg.diag(beta)  # [B,H,C,C]
        T = tf.einsum("bhij,bhjk->bhik", T_left, beta_diag)

        W = tf.einsum("bhij,bhjd->bhid", T, K_beta)
        U = tf.einsum("bhij,bhjd->bhid", T, V_beta)
        return W, U

    def call(self, x):
        # x: [B,T,D]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        beta_logits = self.w_beta(x)  # [B,T,H]

        if self.conv_q is not None:
            q = self.conv_q(q)
            k = self.conv_k(k)
            v = self.conv_v(v)

        q = self._feature_map_with_norm(q)
        k = self._feature_map_with_norm(k)

        # heads
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)
        beta = tf.transpose(beta_logits, [0, 2, 1])
        beta = tf.sigmoid(beta)  # [B,H,T]

        b = tf.shape(q)[0]
        h = tf.shape(q)[1]
        t = tf.shape(q)[2]
        c = self.chunk_size
        num_chunks = (t + c - 1) // c
        pad_len = num_chunks * c - t

        # pad time dim
        def pad_time(z, val=0.0):
            pad = [[0, 0], [0, 0], [0, pad_len], [0, 0]]
            return tf.pad(z, pad, constant_values=val)

        q = pad_time(q)
        k = pad_time(k)
        v = pad_time(v)
        beta = tf.pad(beta, [[0, 0], [0, 0], [0, pad_len]])

        # reshape into chunks [B,H,N,C,Dh]
        new_t = num_chunks * c
        q = tf.reshape(q, [b, h, num_chunks, c, self.d_head])
        k = tf.reshape(k, [b, h, num_chunks, c, self.d_head])
        v = tf.reshape(v, [b, h, num_chunks, c, self.d_head])
        beta = tf.reshape(beta, [b, h, num_chunks, c])

        o = tf.zeros_like(q)
        S = tf.zeros([b, h, self.d_head, self.d_head], dtype=q.dtype)

        causal = tf.linalg.band_part(tf.ones([c, c], dtype=q.dtype), -1, 0)
        causal = tf.reshape(causal, [1, 1, 1, c, c])

        for i in tf.range(num_chunks):
            q_i = q[:, :, i, :, :]
            k_i = k[:, :, i, :, :]
            v_i = v[:, :, i, :, :]
            beta_i = beta[:, :, i, :]

            W_i, U_i = self._compute_ut_transform(k_i, v_i, beta_i)
            pseudo = U_i - tf.einsum("bhcd,bhde->bhce", W_i, S)

            scores = tf.einsum("bhcd,bhce->bhde", q_i, k_i) * causal
            o_intra = tf.einsum("bhde,bhec->bhdc", scores, pseudo)
            o_inter = tf.einsum("bhcd,bhde->bhce", q_i, S)
            o = tf.tensor_scatter_nd_update(
                o,
                indices=tf.reshape(
                    tf.stack(
                        [
                            tf.range(b)[:, None, None, None],
                            tf.range(h)[None, :, None, None],
                            tf.fill([b, h, 1, 1], i),
                            tf.range(c)[None, None, :, None],
                            tf.range(self.d_head)[None, None, None, :],
                        ],
                        axis=-1,
                    ),
                    [b * h * c * self.d_head, 5],
                ),
                updates=tf.reshape(o_intra + o_inter, [b * h * c * self.d_head]),
            )

            delta = pseudo
            S = S + tf.einsum("bhce,bhcf->bhef", delta, k_i)

        # merge heads and project
        o = tf.reshape(o, [b, h, new_t, self.d_head])
        o = o[:, :, :t, :]
        o = self._merge_heads(o)
        return self.wo(o)


class DeltaNetTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads, chunk_size=64, conv_kernel=None, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = DeltaNetAttention(d_model, num_heads, chunk_size, conv_kernel)
        self.act1 = DynamicTanh()
        self.act2 = DynamicTanh()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(d_ff, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ])
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x)
        out1 = self.act1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(out1)
        return self.act2(out1 + self.drop2(ffn_out, training=training))


def build_deltanet_transformer_classifier(
    num_particles,
    feature_dim,
    d_model=16,
    d_ff=16,
    output_dim=5,
    num_heads=4,
    chunk_size=64,
    conv_kernel=None,
    num_layers=1,
    dropout=0.1,
):
    inputs = tf.keras.layers.Input((num_particles, feature_dim))
    x = tf.keras.layers.Dense(d_model, activation="relu")(inputs)
    for _ in range(num_layers):
        x = DeltaNetTransformerBlock(
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            chunk_size=chunk_size,
            conv_kernel=conv_kernel,
            dropout=dropout,
        )(x)
    x = AggregationLayer("max")(x)
    x = tf.keras.layers.Dense(d_model, activation="relu")(x)
    activation = "sigmoid" if output_dim == 1 else "softmax"
    outputs = tf.keras.layers.Dense(output_dim, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)


