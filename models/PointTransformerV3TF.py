import tensorflow as tf
from tensorflow.keras import layers, Model
import math


class AggregationLayer(layers.Layer):
	def __init__(self, aggreg="max", **kwargs):
		super().__init__(**kwargs)
		self.aggreg = aggreg

	def call(self, inputs):
		if self.aggreg == "mean":
			return tf.reduce_mean(inputs, axis=1)
		if self.aggreg == "max":
			return tf.reduce_max(inputs, axis=1)
		raise ValueError("Aggregation must be 'mean' or 'max'.")


# Removed DynamicTanh layer as it's not in the original PTv3.
# We will use standard residual connections.


class CPE1D(layers.Layer):
	def __init__(self, channels, kernel_size=3, **kwargs):
		super().__init__(**kwargs)
		self.channels = channels
		self.kernel_size = kernel_size
		self.depthwise = layers.DepthwiseConv1D(kernel_size=kernel_size, padding="same")
		self.pointwise = layers.Dense(channels)
		self.norm = layers.LayerNormalization(epsilon=1e-6)

	def call(self, x):
		residual = x
		y = self.depthwise(x)
		y = self.pointwise(y)
		y = self.norm(y)
		return residual + y


class RelativePositionalBias(layers.Layer):
	"""
    Calculates relative positional bias from coordinates, not features.
    This aligns with the geometric-aware RPE in the official model.
    """
	def __init__(self, num_heads, hidden=32, **kwargs):
		super().__init__(**kwargs)
		self.num_heads = num_heads
		self.hidden = hidden
		self.mlp = None

	def build(self, input_shape):
		# input_shape is coord_shape [B, T, C_dim]
		coord_dim = input_shape[-1]
		# We will explicitly handle 2D (eta, phi) and 
		# still pass 2D (delta_eta, delta_phi) to the MLP.
		self.mlp = tf.keras.Sequential([
			layers.Dense(self.hidden, activation="gelu"),
			layers.Dense(self.num_heads)
		], name="rpe_mlp")
		super().build(input_shape)

	def call(self, coords):
		# coords shape [B, T, 2] (assumed [eta, phi])
		tf.Assert(tf.shape(coords)[-1] == 2, ["Coordinate dimension must be 2 (eta, phi)"])

		# We want to compute relative positions: [B, T, T, 2]
		coords_i = tf.expand_dims(coords, axis=2)
		coords_j = tf.expand_dims(coords, axis=1)
		
		# Calculate relative eta and phi
		rel_eta = coords_i[..., 0] - coords_j[..., 0]
		rel_phi = coords_i[..., 1] - coords_j[..., 1]

		# --- Handle phi periodicity ---
		# Map rel_phi to the range [-pi, pi]
		pi = tf.constant(math.pi, dtype=coords.dtype)
		rel_phi = tf.math.floormod(rel_phi + pi, 2 * pi) - pi
		# ------------------------------

		# Stack them back together
		rel_coords = tf.stack([rel_eta, rel_phi], axis=-1) # Shape [B, T, T, 2]
		
		# Pass relative coordinates through MLP
		bias = self.mlp(rel_coords)  # [B, T, T, H]
		bias = tf.transpose(bias, [0, 3, 1, 2])  # [B, H, T, T]
		return bias


class PointTransformerAttention(layers.Layer):
	"""
    Modified to perform LOCAL (patched) attention.
    It now accepts a patch_size and reshapes the input sequence.
    It also accepts coordinates to pass to the RPE.
    """
	def __init__(self, d_model, num_heads, patch_size, dropout=0.0, use_rpe=False, **kwargs):
		super().__init__(**kwargs)
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
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
		self.rpe = RelativePositionalBias(num_heads) if use_rpe else None

	@staticmethod
	def _split_heads(x, num_heads):
		b = tf.shape(x)[0]
		t = tf.shape(x)[1]
		d = tf.shape(x)[2]
		dh = d // num_heads
		x = tf.reshape(x, [b, t, num_heads, dh])
		return tf.transpose(x, [0, 2, 1, 3])

	@staticmethod
	def _merge_heads(x):
		b = tf.shape(x)[0]
		t = tf.shape(x)[2]
		h = tf.shape(x)[1]
		dh = tf.shape(x)[3]
		x = tf.transpose(x, [0, 2, 1, 3])
		return tf.reshape(x, [b, t, h * dh])

	def call(self, x, coords, training=False):
		# x shape: [B, T, D]
		# coords shape: [B, T, C_dim]
		
		B, T, D = tf.unstack(tf.shape(x))
		P = self.patch_size
		C = tf.shape(coords)[-1]
		
		# Ensure sequence length is divisible by patch size
		tf.Assert(T % P == 0, [f"Sequence length {T} must be divisible by patch_size {P}"])
		num_patches = T // P

		# Reshape to create patches. This is the dense equivalent of serialization.
		# [B, T, D] -> [B, num_patches, P, D] -> [B * num_patches, P, D]
		x_patched = tf.reshape(x, [B, num_patches, P, D])
		x_patched = tf.reshape(x_patched, [B * num_patches, P, D])

		# Do the same for coordinates
		# [B, T, C] -> [B, num_patches, P, C] -> [B * num_patches, P, C]
		coords_patched = tf.reshape(coords, [B, num_patches, P, C])
		coords_patched = tf.reshape(coords_patched, [B * num_patches, P, C])

		# Now all ops run *within* patches
		q = self._split_heads(self.wq(x_patched), self.num_heads)  # [B*np, H, P, Dh]
		k = self._split_heads(self.wk(x_patched), self.num_heads)
		v = self._split_heads(self.wv(x_patched), self.num_heads)
		
		dk = tf.cast(self.d_head, x.dtype)
		scores = tf.einsum("bhtd,bhTd->bhtT", q, k) / tf.math.sqrt(dk)
		
		if self.use_rpe:
			# RPE also operates on patched coordinates
			bias = self.rpe(coords_patched)  # [B*np, H, P, P]
			scores = scores + bias
			
		weights = tf.nn.softmax(scores, axis=-1)
		weights = self.dropout(weights, training=training)
		
		out = tf.einsum("bhtT,bhTd->bhtd", weights, v)  # [B*np, H, P, Dh]
		out = self._merge_heads(out)  # [B*np, P, D]
		
		# Project and "un-patch"
		out = self.wo(out) # [B*np, P, D]
		out = tf.reshape(out, [B, num_patches, P, D]) # [B, np, P, D]
		out = tf.reshape(out, [B, T, D]) # [B, T, D]
		
		return out


class PTv3Block(layers.Layer):
	"""
    Modified to accept [x, coords] as input.
    Removed DynamicTanh and uses standard residual connections.
    """
	def __init__(self, d_model, d_ff, num_heads, patch_size, cpe_kernel=3, dropout=0.0, use_rpe=False, **kwargs):
		super().__init__(**kwargs)
		self.cpe = CPE1D(d_model, kernel_size=cpe_kernel)
		self.norm1 = layers.LayerNormalization(epsilon=1e-6)
		self.attn = PointTransformerAttention(
			d_model, num_heads, patch_size, dropout=dropout, use_rpe=use_rpe
		)
		self.drop1 = layers.Dropout(dropout)
		self.norm2 = layers.LayerNormalization(epsilon=1e-6)
		self.ffn = tf.keras.Sequential([
			layers.Dense(d_ff, activation="gelu"),
			layers.Dropout(dropout),
			layers.Dense(d_model),
		])
		self.drop2 = layers.Dropout(dropout)
		# Removed DynamicTanh
		
	def call(self, inputs, training=False):
		x, coords = inputs  # Expect a list/tuple of [features, coordinates]
		
		x_res = x
		x = self.cpe(x)  # CPE only on features
		x = x_res + x # Apply residual *after* CPE
		
		# Pass both features and coords to attention
		y = self.attn(self.norm1(x), coords, training=training)
		x = x + self.drop1(y, training=training) # Standard residual
		
		y = self.ffn(self.norm2(x))
		x = x + self.drop2(y, training=training) # Standard residual
		
		return [x, coords] # Pass coords through for the next block


class DownsampleLayer(layers.Layer):
	"""
	A simple downsampling layer using strided 1D convolution
	to create a hierarchy.
	"""
	def __init__(self, out_dim, stride=2, **kwargs):
		super().__init__(**kwargs)
		self.stride = stride
		# Use padding="same" to handle sequence lengths that are not perfectly divisible
		self.conv = layers.Conv1D(
			out_dim, kernel_size=stride, strides=stride, padding="same"
		)
		self.norm = layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		x, coords = inputs
		x = self.conv(x)
		x = self.norm(x)
		
		# Subsample coordinates
		coords = coords[:, ::self.stride, :]
		return [x, coords]


def build_pointtransformer_v3_classifier(
	num_particles,
	feature_dim,
	output_dim=5,
	# Define hierarchical encoder stages
	enc_dims=[64, 128, 256],
	enc_layers=[2, 2, 2],
	enc_heads=[4, 8, 8],
	enc_patch_sizes=[64, 32, 16], # Patch size must be compatible with sequence length
	enc_strides=[2, 2],
	cpe_kernel=3,
	use_rpe=False,
	dropout=0.0,
	aggregation="max",
):
	"""
    Builds a hierarchical, patched Point Transformer for classification.

    Args:
        num_particles (int): Max number of particles (must be divisible by patch sizes).
        feature_dim (int): Input feature dimension. Assumes [pt, eta, phi, ...].
        output_dim (int): Number of output classes.
        enc_dims (list): List of feature dimensions for each encoder stage.
        enc_layers (list): List of num_layers for each encoder stage.
        enc_heads (list): List of num_heads for each encoder stage.
        enc_patch_sizes (list): List of patch sizes for each encoder stage.
        enc_strides (list): List of strides for each DownsampleLayer.
    """
	
	assert len(enc_dims) == len(enc_layers) == len(enc_heads) == len(enc_patch_sizes)
	assert len(enc_dims) == len(enc_strides) + 1
	num_stages = len(enc_dims)

	# Define single input
	features_input = layers.Input((num_particles, feature_dim), name="features")

	# Extract coordinates (eta, phi) from features (assumed indices 1 and 2)
	coords = features_input[..., 1:3] # Shape (B, T, 2)

	# Initial projection on all features
	x = layers.Dense(enc_dims[0], activation="relu")(features_input)

	# --- Hierarchical Encoder ---
	for i in range(num_stages):
		stage_dim = enc_dims[i]
		stage_layers = enc_layers[i]
		stage_heads = enc_heads[i]
		stage_patch_size = enc_patch_sizes[i]
		
		# Check patch size compatibility
		current_seq_len = x.shape[1]
		if current_seq_len is not None:
			if current_seq_len % stage_patch_size != 0:
				raise ValueError(
					f"Stage {i}: Sequence length {current_seq_len} is not divisible by patch size {stage_patch_size}"
				)
		else:
			# Dynamic shape check
			x_shape = tf.shape(x)
			tf.Assert(x_shape[1] % stage_patch_size == 0, [
				f"Stage {i}: Dynamic sequence length", x_shape[1], "is not divisible by patch size", stage_patch_size
			])


		# Add transformer blocks
		for _ in range(stage_layers):
			x, coords = PTv3Block(
				d_model=stage_dim,
				d_ff=stage_dim * 4, # Common to use 4*d_model for d_ff
				num_heads=stage_heads,
				patch_size=stage_patch_size,
				cpe_kernel=cpe_kernel,
				dropout=dropout,
				use_rpe=use_rpe,
			)([x, coords])
		
		# Downsample (if not the last stage)
		if i < num_stages - 1:
			x, coords = DownsampleLayer(
				out_dim=enc_dims[i+1],
				stride=enc_strides[i]
			)([x, coords])

	# --- Classifier Head ---
	x = AggregationLayer(aggregation)(x)
	x = layers.Dense(enc_dims[-1], activation="relu")(x) # Final MLP
	
	activation = "sigmoid" if output_dim == 1 else "softmax"
	outputs = layers.Dense(output_dim, activation=activation)(x)
	
	return Model(inputs=features_input, outputs=outputs)