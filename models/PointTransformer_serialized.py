import tensorflow as tf
from tensorflow.keras import layers, Model
import math


# ========== Core Components ==========

class Serialization2D(layers.Layer):
    """
    Serialization and sorting for point sequences.
    - 'morton': 2D Z-order (Morton) on (eta, phi) grid
    - 'pt':     sort by pt descending
    - 'kt':     sort by pt * sqrt(eta^2 + phi^2) descending
    Inputs:
      x:      features [B, N, C]
      coords: either [B, N, 2] = (eta, phi) or [B, N, 3] = (pt, eta, phi)
    """
    def __init__(self, grid_size=0.05, sort_by="morton", **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.sort_by = sort_by
        
    def _interleave_bits(self, x, y):
        """A simplified Z-order curve (Morton code) for 2D quantization."""
        x = tf.cast(x, tf.int64)
        y = tf.cast(y, tf.int64)
        z = tf.zeros_like(x)

        one = tf.constant(1, dtype=tf.int64)
        # Interleave bits (up to 30 bits for common precision); ensure int64 loop var
        for i in tf.range(30, dtype=tf.int64):
            mask_x = (tf.bitwise.right_shift(x, i) & one)
            mask_y = (tf.bitwise.right_shift(y, i) & one)

            z = z | tf.bitwise.left_shift(mask_x, 2 * i)
            z = z | tf.bitwise.left_shift(mask_y, 2 * i + 1)
        return z

    def call(self, inputs):
        # Support both call(x, coords) and call([x, coords])
        if isinstance(inputs, (list, tuple)):
            x, coords = inputs
        else:
            raise ValueError("Serialization2D expects inputs=[x, coords]")
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        last_dim = tf.shape(coords)[-1]
        # coords can be (eta, phi) or (pt, eta, phi)
        if coords.shape.rank is None or coords.shape.rank != 3:
            raise ValueError("coords must be rank-3: [B, N, 2 or 3]")

        # Decide sorting strategy
        sort_by = tf.convert_to_tensor(self.sort_by)
        # Default: morton
        if self.sort_by == "morton":
            eta = coords[..., -2]  # works for both 2 or 3 inputs
            phi = coords[..., -1]
            # Quantize to grid (min-aligned)
            eta_min = tf.reduce_min(eta, axis=1, keepdims=True)
            phi_min = tf.reduce_min(phi, axis=1, keepdims=True)
            grid_eta = tf.cast((eta - eta_min) / self.grid_size, tf.int32)
            grid_phi = tf.cast((phi - phi_min) / self.grid_size, tf.int32)
            morton_code = self._interleave_bits(grid_eta, grid_phi)
            sort_idx = tf.argsort(morton_code, axis=1)  # ascending
        elif self.sort_by == "pt":
            if coords.shape[-1] < 3:
                raise ValueError("pt sorting requires coords[...,0]=pt with coords last-dim=3")
            pt = coords[..., 0]
            # sort by pt descending
            sort_idx = tf.argsort(pt, axis=1, direction="DESCENDING")
        elif self.sort_by == "kt":
            if coords.shape[-1] < 3:
                raise ValueError("kt sorting requires coords[...,0:3]=(pt,eta,phi)")
            pt = coords[..., 0]
            eta = coords[..., 1]
            phi = coords[..., 2]
            kt = pt * tf.sqrt(eta * eta + phi * phi)
            sort_idx = tf.argsort(kt, axis=1, direction="DESCENDING")
        else:
            raise ValueError(f"Unknown sort_by: {self.sort_by}")

        # Prepare indices for gathering
        batch_idx = tf.range(B)[:, None]
        batch_idx = tf.tile(batch_idx, [1, N])
        gather_idx = tf.stack([batch_idx, sort_idx], axis=-1)

        x_sorted = tf.gather_nd(x, gather_idx)
        coords_sorted = tf.gather_nd(coords, gather_idx)
        return x_sorted, coords_sorted

class GeometricCPE(layers.Layer):
    """
    Convolutional Position Encoding (xCPE) that respects jet geometry.
    Uses 2D convolution on (eta, phi) grid.
    This serves as the efficient positional injection mechanism in PTv3.
    """
    def __init__(self, channels, kernel_size=3, grid_size=0.05, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        
        # 2D conv on spatial grid (mimicking sparse conv)
        self.conv2d = layers.Conv2D(
            channels, 
            kernel_size=kernel_size, 
            padding="same",
            groups=channels,  # Depthwise convolution for efficiency
            use_bias=True
        )
        self.pointwise = layers.Dense(channels)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, eta, phi):
        """
        Args:
            x: Features [B, N, C]
            eta: Particle eta [B, N]
            phi: Particle phi [B, N]
        """
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        C = self.channels
        
        residual = x
        
        # Quantize to grid
        eta_min = tf.reduce_min(eta, axis=1, keepdims=True)
        phi_min = tf.reduce_min(phi, axis=1, keepdims=True)
        
        grid_eta = tf.cast((eta - eta_min) / self.grid_size, tf.int32)
        grid_phi = tf.cast((phi - phi_min) / self.grid_size, tf.int32)
        
        # Get grid dimensions
        H = tf.reduce_max(grid_eta) + 1
        W = tf.reduce_max(grid_phi) + 1
        
        # Scatter particles onto 2D grid [B, H, W, C]
        batch_idx = tf.range(B)[:, None]
        batch_idx = tf.tile(batch_idx, [1, N])
        
        indices = tf.stack([
            batch_idx,
            grid_eta,
            grid_phi
        ], axis=-1)  # [B, N, 3]
        
        indices = tf.reshape(indices, [-1, 3])
        features = tf.reshape(x, [-1, C])
        
        # Create a sparse tensor equivalent
        grid = tf.scatter_nd(
            indices,
            features,
            [B, H, W, C]
        )
        
        # Apply 2D convolution
        grid = self.conv2d(grid)
        
        # Gather back to particles
        out = tf.gather_nd(grid, tf.reshape(indices, [B, N, 3]))
        
        out = self.pointwise(out)
        out = self.norm(out)
        
        return residual + out

class QuantizedRPE(layers.Layer):
    """
    RPE using quantized relative positions with learnable table (reintroduced if needed).
    """
    def __init__(self, num_heads, quantization_bins=32, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.bins = quantization_bins
        
        # Learnable bias table for quantized relative positions
        # Each dimension (eta, phi) gets its own set of bins
        self.rpe_table = self.add_weight(
            name="rpe_table",
            shape=[2 * quantization_bins, num_heads],
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
    
    def call(self, coords):
        """
        Args:
            coords: [B, T, 2] where coords[..., 0] = eta, coords[..., 1] = phi
        Returns:
            bias: [B, H, T, T]
        """
        eta = coords[..., 0]  # [B, T]
        phi = coords[..., 1]  # [B, T]
        
        # Compute relative positions
        rel_eta = eta[:, :, None] - eta[:, None, :]  # [B, T, T]
        rel_phi = phi[:, :, None] - phi[:, None, :]
        
        # Handle phi periodicity
        pi = tf.constant(math.pi, dtype=phi.dtype)
        rel_phi = tf.math.floormod(rel_phi + pi, 2 * pi) - pi
        
        # Quantize to bins
        # Map to [-bins/2, bins/2] range
        eta_range = tf.reduce_max(tf.abs(rel_eta))
        phi_range = tf.constant(math.pi, dtype=phi.dtype)
        
        # Avoid division by zero
        eta_range = tf.maximum(eta_range, 1e-6)
        
        eta_bins = tf.cast(
            rel_eta / eta_range * (self.bins // 2), 
            tf.int32
        )
        phi_bins = tf.cast(
            rel_phi / phi_range * (self.bins // 2),
            tf.int32
        )
        
        # Clamp to valid range
        eta_bins = tf.clip_by_value(eta_bins, -self.bins // 2, self.bins // 2 - 1)
        phi_bins = tf.clip_by_value(phi_bins, -self.bins // 2, self.bins // 2 - 1)
        
        # Shift to positive indices
        eta_idx = eta_bins + self.bins // 2
        phi_idx = phi_bins + self.bins // 2 + self.bins  # Offset for second dimension
        
        # Lookup in table
        eta_bias = tf.gather(self.rpe_table, eta_idx)  # [B, T, T, H]
        phi_bias = tf.gather(self.rpe_table, phi_idx)
        
        # Combine biases
        bias = eta_bias + phi_bias  # [B, T, T, H]
        bias = tf.transpose(bias, [0, 3, 1, 2])  # [B, H, T, T]
        
        return bias


class PatchedAttention(layers.Layer):
    """Local attention with patching on the serialized sequence."""
    def __init__(self, d_model, num_heads, patch_size, dropout=0.0, use_rpe=False, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.patch_size = patch_size
        self.use_rpe = use_rpe
        
        self.wq = layers.Dense(d_model, use_bias=True)
        self.wk = layers.Dense(d_model, use_bias=True)
        self.wv = layers.Dense(d_model, use_bias=True)
        self.wo = layers.Dense(d_model, use_bias=True)
        self.dropout = layers.Dropout(dropout)
        self.rpe = QuantizedRPE(num_heads) if use_rpe else None
    
    def _split_heads(self, x, num_heads):
        b, t, d = tf.unstack(tf.shape(x)[:3])
        x = tf.reshape(x, [b, t, num_heads, d // num_heads])
        return tf.transpose(x, [0, 2, 1, 3])
    
    def _merge_heads(self, x):
        b, h, t, dh = tf.unstack(tf.shape(x))
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, t, h * dh])
    
    def call(self, x, coords, training=False):
        B, T, D = tf.unstack(tf.shape(x))
        P = self.patch_size
        
        # Pad if necessary
        pad_len = (P - T % P) % P
        if pad_len > 0:
            x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
            coords = tf.pad(coords, [[0, 0], [0, pad_len], [0, 0]])
        
        T_padded = T + pad_len
        num_patches = T_padded // P
        
        # Reshape to patches
        x_patched = tf.reshape(x, [B, num_patches, P, D])
        x_patched = tf.reshape(x_patched, [B * num_patches, P, D])
        coords_patched = tf.reshape(coords, [B, num_patches, P, 2])
        coords_patched = tf.reshape(coords_patched, [B * num_patches, P, 2])
        
        # Attention within patches
        q = self._split_heads(self.wq(x_patched), self.num_heads)
        k = self._split_heads(self.wk(x_patched), self.num_heads)
        v = self._split_heads(self.wv(x_patched), self.num_heads)
        
        dk = tf.cast(self.d_head, x.dtype)
        scores = tf.einsum("bhtd,bhTd->bhtT", q, k) / tf.math.sqrt(dk)
        
        if self.use_rpe:
            bias = self.rpe(coords_patched)
            scores = scores + bias
        
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        
        out = tf.einsum("bhtT,bhTd->bhtd", weights, v)
        out = self._merge_heads(out)
        out = tf.ensure_shape(out, [None, None, self.d_model])
        out = self.wo(out)
        
        # Reshape back
        out = tf.reshape(out, [B, num_patches, P, self.d_model])
        out = tf.reshape(out, [B, T_padded, self.d_model])
        
        # Remove padding
        if pad_len > 0:
            out = out[:, :-pad_len, :]
        
        return out


class PTv3Block(layers.Layer):
    """Transformer block with CPE."""
    def __init__(self, d_model, d_ff, num_heads, patch_size, cpe_k=3, grid_size=0.05, dropout=0.0, use_rpe=False, **kwargs):
        super().__init__(**kwargs)
        self.cpe = GeometricCPE(d_model, kernel_size=cpe_k, grid_size=grid_size)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = PatchedAttention(d_model, num_heads, patch_size, dropout=dropout, use_rpe=use_rpe)
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout)
    
    def call(self, inputs, training=False):
        x, coords = inputs
        eta, phi = coords[..., 0], coords[..., 1]
        
        # CPE
        x = self.cpe(x, eta, phi)
        
        # Attention
        y = self.attn(self.norm1(x), coords, training=training)
        x = x + self.drop1(y, training=training)
        
        # FFN
        y = self.ffn(self.norm2(x))
        x = x + self.drop2(y, training=training)
        
        return [x, coords]


class SerializedPooling2D(layers.Layer):
    """
    Pools along the serialized 1D sequence (Grid Pooling approximation).
    The input sequence is assumed to be already sorted by the serialization code.
    """
    def __init__(self, out_dim, stride=2, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.proj = layers.Dense(out_dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs):
        x, coords = inputs
        B, N = tf.shape(x)[0], tf.shape(x)[1]
        channels = x.shape[-1]
        coord_channels = tf.shape(coords)[-1]
        
        # Pad and group (This is now simpler as it just pads the sorted sequence)
        N_out = N // self.stride
        remainder = N % self.stride
        if remainder != 0:
            pad_len = self.stride - remainder
            # Pad at the end of the sequence
            x = tf.pad(x, [[0, 0], [0, pad_len], [0, 0]])
            coords = tf.pad(coords, [[0, 0], [0, pad_len], [0, 0]])
            N_out = (N + pad_len) // self.stride
        
        # Reshape for pooling
        x_grouped = tf.reshape(x, [B, N_out, self.stride, channels])
        coords_grouped = tf.reshape(coords, [B, N_out, self.stride, coord_channels])
        
        # Pool (Max features, Mean coordinates)
        x_pooled = tf.reduce_max(x_grouped, axis=2)
        coords_pooled = tf.reduce_mean(coords_grouped, axis=2)
        
        # Project and Normalize
        x_pooled = tf.ensure_shape(x_pooled, [None, None, channels])
        x_pooled = self.proj(x_pooled)
        x_pooled = self.norm(x_pooled)
        
        return [x_pooled, coords_pooled]


# ========== Full Model ==========

def build_ptv3_serialized_jet_classifier(
    num_particles=150,
    output_dim=5,
    enc_dims=[64, 128, 256],
    enc_layers=[2, 2, 2],
    enc_heads=[4, 8, 8],
    enc_patch_sizes=[50, 25, 12],
    enc_strides=[3, 2],
    cpe_k=3, # Changed default to 3x3 for simpler conv, was 8
    grid_size=0.05,
    use_rpe=False, # Changed default to False, aligning with PTv3 principle
    dropout=0.0,
    aggregation="max",
    serialize_by="morton",
    use_pool=True,
):
    """Build hierarchical PTv3-inspired jet classifier with serialization."""
    
    # Input: [pt, eta, phi]
    features_input = layers.Input((num_particles, 3), name="features")
    
    # Extract coordinates
    # Pass full triplet for flexible sorting (pt, eta, phi)
    coords = features_input[..., 0:3]
    
    # Initial projection
    x = layers.Dense(enc_dims[0], activation="relu")(features_input)
    
    # ------------------ START Serialization Injection ------------------
    # Step 1: Serialize the initial point cloud.
    # All subsequent layers (Blocks and Pooling) operate on this sorted sequence.
    # Choose sorting with sort_by: "morton" (default), "pt", or "kt"
    x, coords = Serialization2D(grid_size=grid_size, sort_by=serialize_by)([x, coords])
    # ------------------- END Serialization Injection -------------------
    
    # Hierarchical encoder
    for i in range(len(enc_dims)):
        # Transformer blocks
        for _ in range(enc_layers[i]):
            x, coords = PTv3Block(
                d_model=enc_dims[i],
                d_ff=enc_dims[i] * 4,
                num_heads=enc_heads[i],
                patch_size=enc_patch_sizes[i],
                cpe_k=cpe_k,
                grid_size=grid_size,
                dropout=dropout,
                use_rpe=use_rpe,
            )([x, coords])
        
        # Downsample (except last stage)
        if i < len(enc_dims) - 1:
            if use_pool:
                # Downsampling uses SerializedPooling2D
                x, coords = SerializedPooling2D(
                    out_dim=enc_dims[i + 1],
                    stride=enc_strides[i]
                )([x, coords])
            else:
                # No pooling: project channels only, keep sequence length and coords
                x = layers.Dense(enc_dims[i + 1])(x)
                x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # NOTE: In the official PTv3, SerializedPooling calculates the new,
            # coarser serialization codes and implicitly sorts the pooled points.
            # This implementation relies on the pooling operation's local grouping
            # (due to the prior Z-order sort) to maintain the spatial structure.
    
    # Aggregation
    if aggregation == "mean":
        x = tf.reduce_mean(x, axis=1)
    else:
        x = tf.reduce_max(x, axis=1)
    
    # Classifier head
    x = layers.Dense(enc_dims[-1], activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    
    activation = "sigmoid" if output_dim == 1 else "softmax"
    outputs = layers.Dense(output_dim, activation=activation)(x)
    
    return Model(inputs=features_input, outputs=outputs)


# Example usage
if __name__ == "__main__":
    model = build_ptv3_serialized_jet_classifier(
        num_particles=150,
        output_dim=5,
        enc_dims=[64, 128, 256],
        enc_layers=[2, 2, 2],
        enc_heads=[4, 8, 8],
        enc_patch_sizes=[50, 25, 12],
        enc_strides=[3, 2],
        cpe_k=3,
        use_rpe=False,
        dropout=0.1,
        aggregation="max",
        serialize_by="morton"
    )
    
    model.summary()
    
    # Test forward pass
    batch_size = 4
    dummy_input = tf.random.normal([batch_size, 150, 3])
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")