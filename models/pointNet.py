import tensorflow as tf
from tensorflow.keras import layers, Model


def _conv_bn_relu(x, filters, kernel_size, name):
	conv = layers.Conv2D(
		filters,
		kernel_size=kernel_size,
		strides=(1, 1),
		padding="valid",
		use_bias=False,
		name=f"{name}_conv",
	)(x)
	bn = layers.BatchNormalization(name=f"{name}_bn")(conv)
	return layers.Activation("relu", name=f"{name}_relu")(bn)


def _dense_bn_relu(x, units, name):
	d = layers.Dense(units, use_bias=False, name=f"{name}_dense")(x)
	bn = layers.BatchNormalization(name=f"{name}_bn")(d)
	return layers.Activation("relu", name=f"{name}_relu")(bn)


def build_pointnet_classifier(
	num_particles,
	feature_dim=3,
	output_dim=5,
	dropout_rate=0.3,
):
	"""
	PointNet classifier adapted for jet tagging.
	- Input: (batch, num_particles, feature_dim) where feature_dim=3 for (pt, eta, phi)
	- Output: (batch, output_dim)
	"""
	inputs = layers.Input((num_particles, feature_dim), name="features")

	# Add a channel dimension to match Conv2D expectations: (B, N, F, 1)
	x = layers.Lambda(lambda t: tf.expand_dims(t, axis=-1), name="expand_dims")(inputs)

	# Point functions (MLPs implemented as 2D convs with kernel (1, F) then (1,1))
	# First conv spans the feature dimension (F), subsequent are pointwise (1,1)
	x = _conv_bn_relu(x, 16, (1, feature_dim), name="conv1")  # (B, N, 1, 64)
	x = _conv_bn_relu(x, 32, (1, 1), name="conv2")
	x = _conv_bn_relu(x, 64, (1, 1), name="conv3")
	# x = _conv_bn_relu(x, 128, (1, 1), name="conv4")
	# x = _conv_bn_relu(x, 1024, (1, 1), name="conv5")

	# Symmetric function over points: max pooling across the N (particle) dimension
	# After first conv, width dimension is 1, so GlobalMaxPooling2D is equivalent to max over N.
	x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)

	# MLP on global feature
	x = _dense_bn_relu(x, 32, name="fc1")
	x = _dense_bn_relu(x, 16, name="fc2")
	x = layers.Dropout(dropout_rate, name="dropout")(x)

	activation = "sigmoid" if output_dim == 1 else "softmax"
	outputs = layers.Dense(output_dim, activation=activation, name="head")(x)
	return Model(inputs, outputs, name="PointNetJetClassifier")


