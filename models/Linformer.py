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
# Linformer Multi-Head Attention
# ---------------------------
class LinformerMultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, proj_dim, **kwargs):
        """
        Args:
          d_model: Dimensionality of the model.
          num_heads: Number of attention heads.
          proj_dim: The projection dimension to which keys and values will be reduced.
        """
        super(LinformerMultiHeadAttention, self).__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.proj_dim = proj_dim

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, d_model)
        self.seq_len = input_shape[1]
        # Standard dense weight matrices for Q, K, and V.
        self.wq = self.add_weight(shape=(self.d_model, self.d_model),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  name="wq")
        self.wk = self.add_weight(shape=(self.d_model, self.d_model),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  name="wk")
        self.wv = self.add_weight(shape=(self.d_model, self.d_model),
                                  initializer='glorot_uniform',
                                  trainable=True,
                                  name="wv")
        self.dense = layers.Dense(self.d_model)
        # Learnable projection matrices for keys and values.
        # These project along the sequence dimension from seq_len -> proj_dim.
        self.E = self.add_weight(shape=(self.num_heads, self.seq_len, self.proj_dim),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name="proj_E")
        self.F = self.add_weight(shape=(self.num_heads, self.seq_len, self.proj_dim),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name="proj_F")
        super(LinformerMultiHeadAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        # x shape: (batch_size, seq_len, d_model)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # Transpose to shape: (batch_size, num_heads, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Compute linear projections.
        q = tf.matmul(x, self.wq)  # (batch_size, seq_len, d_model)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)

        # Split into multiple heads.
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Project keys and values along the sequence dimension.
        # Using Einstein summation: for each head h,
        #   k_proj[b, h] = E[h]^T (applied on the sequence dim of k[b, h])
        # k_proj shape: (batch_size, num_heads, proj_dim, depth)
        k_proj = tf.einsum('bhnd, hnr -> bhrd', k, self.E)
        v_proj = tf.einsum('bhnd, hnr -> bhrd', v, self.F)

        # Scaled dot-product attention.
        # Compute scores between queries and projected keys.
        # scores shape: (batch_size, num_heads, seq_len, proj_dim)
        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k_proj, transpose_b=True) / tf.math.sqrt(dk)
        attn_weights = tf.nn.softmax(scores, axis=-1)

        # Compute the attention output.
        # Output shape: (batch_size, num_heads, seq_len, depth)
        attn_output = tf.matmul(attn_weights, v_proj)

        # Concatenate heads.
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(attn_output, (batch_size, -1, self.d_model))
        output = self.dense(concat_output)
        return output

# ---------------------------
# Physical Linformer Multi-Head Attention
# ---------------------------
class ClusteredLinformerAttention(layers.Layer):
    def __init__(
        self,
        d_model,
        num_heads,
        proj_dim,
        cluster_E=False,
        cluster_F=False,
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

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, d_model)
        self.seq_len = input_shape[1]
        # Q, K, V weight matrices
        self.wq = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="wq",
        )
        self.wk = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="wk",
        )
        self.wv = self.add_weight(
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
            name="wv",
        )

        # Projection for Keys
        if not self.cluster_E:
            self.E = self.add_weight(
                shape=(self.num_heads, self.seq_len, self.proj_dim),
                initializer="glorot_uniform",
                trainable=True,
                name="proj_E",
            )
        else:
            self.chunk_size_E = (
                self.seq_len + self.proj_dim - 1
            ) // self.proj_dim
            self.cluster_E_W = self.add_weight(
                shape=(self.num_heads, self.proj_dim, self.chunk_size_E),
                initializer="glorot_uniform",
                trainable=True,
                name="cluster_E_weights",
            )

        # Projection for Values
        if not self.cluster_F:
            self.F = self.add_weight(
                shape=(self.num_heads, self.seq_len, self.proj_dim),
                initializer="glorot_uniform",
                trainable=True,
                name="proj_F",
            )
        else:
            self.chunk_size_F = (
                self.seq_len + self.proj_dim - 1
            ) // self.proj_dim
            self.cluster_F_W = self.add_weight(
                shape=(self.num_heads, self.proj_dim, self.chunk_size_F),
                initializer="glorot_uniform",
                trainable=True,
                name="cluster_F_weights",
            )

        # final dense
        self.dense = layers.Dense(self.d_model)
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        # x: (batch, seq_len, d_model) -> (batch, heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Linear projections
        q = tf.matmul(x, self.wq)  # (B, N, d_model)
        k = tf.matmul(x, self.wk)
        v = tf.matmul(x, self.wv)
        # Split heads
        q = self.split_heads(q, batch_size)  # (B, H, N, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Determine projection lengths
        proj_k = self.proj_dim
        proj_v = self.proj_dim

        # Project Keys
        if not self.cluster_E:
            k_proj = tf.einsum("bhnd,hnr->bhrd", k, self.E)
        else:
            pad_k = self.chunk_size_E * self.proj_dim - self.seq_len
            k_p = tf.pad(k, [[0, 0], [0, 0], [0, pad_k], [0, 0]])
            k_chunks = tf.reshape(
                k_p,
                (
                    batch_size,
                    self.num_heads,
                    self.proj_dim,
                    self.chunk_size_E,
                    self.depth,
                ),
            )  # (B,H,C,l,d)
            k_proj = tf.einsum("bhcld,hcl->bhcd", k_chunks, self.cluster_E_W)

        # Project Values
        if not self.cluster_F:
            v_proj = tf.einsum("bhnd,hnr->bhrd", v, self.F)
        else:
            pad_v = self.chunk_size_F * self.proj_dim - self.seq_len
            v_p = tf.pad(v, [[0, 0], [0, 0], [0, pad_v], [0, 0]])
            v_chunks = tf.reshape(
                v_p,
                (
                    batch_size,
                    self.num_heads,
                    self.proj_dim,
                    self.chunk_size_F,
                    self.depth,
                ),
            )
            v_proj = tf.einsum("bhcld,hcl->bhcd", v_chunks, self.cluster_F_W)

        # Scaled dot-product attention
        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k_proj, transpose_b=True) / tf.math.sqrt(dk)
        attn_weights = tf.nn.softmax(scores, axis=-1)
        attn_output = tf.matmul(attn_weights, v_proj)

        # Merge heads
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # (B, N, H, depth)
        concat = tf.reshape(attn_output, (batch_size, -1, self.d_model))
        return self.dense(concat)

# ---------------------------
# Linformer Transformer Block
# ---------------------------
class LinformerTransformerBlock(layers.Layer):
    def __init__(self, d_model, d_ff, output_dim, num_heads, proj_dim, cluster_E=False, cluster_F=False, **kwargs):
        super(LinformerTransformerBlock, self).__init__(**kwargs)
        self.attention = ClusteredLinformerAttention(d_model, num_heads, proj_dim, cluster_E, cluster_F)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])
        self.outD = layers.Dense(output_dim)

    def call(self, x):
        attn_output = self.attention(x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# ---------------------------
# Linformer Transformer Classifier Model
# ---------------------------
def build_linformer_transformer_classifier(num_particles, feature_dim,
                                             d_model=16, d_ff=16, output_dim=16,
                                             num_heads=8, proj_dim=8, cluster_E=False, cluster_F=False):
    """
    Builds a classifier model with:
      - A linear embedding layer.
      - Multiple Linformer transformer blocks.
      - Aggregation over the sequence dimension.
      - A final linear output layer for 5 classes.
    """
    inputs = layers.Input(shape=(num_particles, feature_dim))
    
    x = layers.Dense(d_model, activation='relu')(inputs)
    
    x = LinformerTransformerBlock(d_model, d_ff, output_dim, num_heads, proj_dim, cluster_E, cluster_F)(x)    
    
    pooled_output = AggregationLayer(aggreg='max')(x)
    
    x = layers.Dense(d_model, activation='relu')(pooled_output)
    outputs = layers.Dense(5, activation='softmax')(x)
    
    return Model(inputs=inputs, outputs=outputs)