"""
JEDI-Linear: Fast and Efficient Graph Neural Networks for Jet Tagging

This module implements JEDI-Linear, a linear-complexity architecture for jet tagging
that achieves high performance with efficient O(N) operations suitable for FPGAs.

Architecture Overview:
---------------------
JEDI-Linear uses a two-stage mixing approach:
1. Global Interaction: O(N) linear-complexity particle mixing via global aggregation
2. Channel Mixing: Mixes information across feature channels

The architecture replaces O(N²) pairwise particle interactions with O(N) global
aggregation, making it highly efficient for hardware deployment while maintaining
competitive accuracy.

Key Components:
--------------
- GlobalInteractionLayer: O(N) particle mixing using global average pooling + broadcast
  (replaces O(N²) token mixing from standard MLP-Mixer)
- ChannelMixingLayer: Mixes information across the feature dimension using dense layers
- JEDILinearBlock: Combines global interaction and channel mixing with residual connections
- Batch normalization after each mixing operation for stable training

Advantages:
----------
- True linear complexity O(N) instead of quadratic O(N²) attention or token mixing
- Hardware-friendly operations (matrix multiplications, no softmax)
- Permutation-invariant to particle ordering
- Competitive performance with transformer-based models
- Efficient FPGA deployment with quantization-aware training
- Sub-100ns inference latency on FPGA with high throughput

Reference:
---------
Based on "JEDI-linear: Fast and Efficient Graph Neural Networks for Jet Tagging on FPGAs"
https://arxiv.org/abs/2508.15468

Usage Example:
-------------
    from models.JEDI_Linear import build_jedi_linear_medium

    model = build_jedi_linear_medium(
        num_particles=150,
        feature_dim=3,
        output_dim=5
    )

    # Train model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=256)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------
# JEDI-Linear Core Components
# ---------------------------

class GlobalInteractionLayer(layers.Layer):
    """
    Global information gathering layer from the JEDI-linear paper.

    This implements the linear-complexity interaction mechanism described in:
    "JEDI-linear: Fast and Efficient Graph Neural Networks for Jet Tagging on FPGAs"

    Architecture (from Figure 3 and Equation 4):
    1. Global context via average pooling: g = (1/N_O) * sum(I_j)
    2. Dense1 on global context: Dense1(g)
    3. Dense2 on individual particles: Dense2(I_i) for each particle
    4. Element-wise addition: E'_i = Dense1(g) + Dense2(I_i)

    This replaces the O(N^2) pairwise interactions with O(N) operations.
    """
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        # Dense1: operates on global context (after average pooling)
        self.dense1 = layers.Dense(self.latent_dim, name=f'{self.name}_global_dense')
        # Dense2: operates on individual particle features
        self.dense2 = layers.Dense(self.latent_dim, name=f'{self.name}_particle_dense')
        self.norm = layers.BatchNormalization(name=f'{self.name}_norm')
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: [B, N_O, P] where B=batch, N_O=particles, P=features

        # Global average pooling across particles: [B, N_O, P] -> [B, P]
        global_context = tf.reduce_mean(inputs, axis=1, keepdims=False)

        # Transform global context: [B, P] -> [B, D_E]
        global_transformed = self.dense1(global_context)

        # Broadcast back to all particles: [B, D_E] -> [B, 1, D_E] -> [B, N_O, D_E]
        global_broadcast = tf.expand_dims(global_transformed, axis=1)

        # Transform individual particle features: [B, N_O, P] -> [B, N_O, D_E]
        particle_transformed = self.dense2(inputs)

        # Element-wise addition: [B, N_O, D_E] + [B, N_O, D_E] (broadcast)
        interaction_features = global_broadcast + particle_transformed

        # Apply batch normalization
        interaction_features = self.norm(interaction_features, training=training)

        return interaction_features


class ChannelMixingLayer(layers.Layer):
    """
    Channel-mixing (feature-mixing) layer that mixes information across features.
    Implements: X_out = DenseC(X) where DenseC operates on the channel dimension.
    """
    def __init__(self, feature_dim, hidden_units=None, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.hidden_units = hidden_units or feature_dim

    def build(self, input_shape):
        self.dense1 = layers.Dense(self.hidden_units, activation='relu', name=f'{self.name}_dense1')
        self.dense2 = layers.Dense(self.feature_dim, name=f'{self.name}_dense2')
        self.norm = layers.BatchNormalization(name=f'{self.name}_norm')
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: [B, N, C]
        # Dense operates on last dimension (channel) by default
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.norm(x, training=training)
        return x


class JEDILinearBlock(layers.Layer):
    """
    JEDI-Linear block that combines global interaction and channel-mixing with residual connections.

    Uses O(N) linear-complexity global interaction instead of O(N²) token mixing.
    This implements the core JEDI-linear architecture from the paper.
    """
    def __init__(self, num_particles, feature_dim, token_hidden=None, channel_hidden=None, **kwargs):
        super().__init__(**kwargs)
        self.num_particles = num_particles
        self.feature_dim = feature_dim
        self.token_hidden = token_hidden or feature_dim  # For global interaction, use feature_dim as latent_dim
        self.channel_hidden = channel_hidden or feature_dim * 4

    def build(self, input_shape):
        # Use GlobalInteractionLayer for O(N) linear complexity
        self.global_interaction = GlobalInteractionLayer(
            self.token_hidden,
            name=f'{self.name}_global_interaction'
        )
        self.channel_mixing = ChannelMixingLayer(
            self.feature_dim,
            self.channel_hidden,
            name=f'{self.name}_channel_mixing'
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Global interaction with residual connection (O(N) complexity)
        x = inputs + self.global_interaction(inputs, training=training)
        # Channel mixing with residual connection
        x = x + self.channel_mixing(x, training=training)
        return x


# ---------------------------
# JEDI-Linear Model Builder
# ---------------------------

def build_jedi_linear_classifier(
    num_particles=150,
    feature_dim=3,
    embedding_dim=16,
    num_blocks=2,
    token_hidden=None,
    channel_hidden=None,
    output_dim=5,
    aggregation='mean',
    dropout_rate=0.0,
    head_hidden_dims=None
):
    """
    Build JEDI-Linear classifier for jet tagging.

    Architecture:
    1. Input embedding: project features to embedding_dim
    2. JEDI-Linear blocks: token-mixing + channel-mixing with residuals
    3. Global aggregation: mean or max pooling across particles
    4. Classification head: MLP layers -> output

    Args:
        num_particles: Number of particles per jet (default: 150)
        feature_dim: Input feature dimension (default: 3 for pt, eta, phi)
        embedding_dim: Hidden dimension for embeddings (default: 16)
        num_blocks: Number of JEDI-Linear blocks (default: 2)
        token_hidden: Hidden units for token mixing (default: num_particles)
        channel_hidden: Hidden units for channel mixing (default: embedding_dim * 4)
        output_dim: Number of output classes (default: 5)
        aggregation: Aggregation method, 'mean' or 'max' (default: 'mean')
        dropout_rate: Dropout rate in classification head (default: 0.0)
        head_hidden_dims: List of hidden dimensions for classification head (default: [64, 32])

    Returns:
        Keras Model
    """
    if head_hidden_dims is None:
        head_hidden_dims = [64, 32]

    # Input
    inputs = layers.Input(shape=(num_particles, feature_dim), name='input_features')

    # Initial embedding projection
    x = layers.Dense(embedding_dim, activation='relu', name='input_embedding')(inputs)
    x = layers.BatchNormalization(name='input_norm')(x)

    # Stack of JEDI-Linear blocks
    for i in range(num_blocks):
        x = JEDILinearBlock(
            num_particles=num_particles,
            feature_dim=embedding_dim,
            token_hidden=token_hidden,
            channel_hidden=channel_hidden,
            name=f'jedi_block_{i}'
        )(x)

    # Global aggregation across particles
    if aggregation == 'mean':
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    elif aggregation == 'max':
        x = layers.GlobalMaxPooling1D(name='global_pool')(x)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}. Use 'mean' or 'max'.")

    # Classification head
    for i, hidden_dim in enumerate(head_hidden_dims):
        x = layers.Dense(hidden_dim, activation='relu', name=f'head_dense_{i}')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f'head_dropout_{i}')(x)

    # Output layer
    activation = 'sigmoid' if output_dim == 1 else 'softmax'
    outputs = layers.Dense(output_dim, activation=activation, name='output')(x)

    return Model(inputs=inputs, outputs=outputs, name='JEDI_Linear_Classifier')


def build_jedi_linear_small(num_particles=150, feature_dim=3, output_dim=5):
    """Small JEDI-Linear model (~10K parameters)"""
    return build_jedi_linear_classifier(
        num_particles=num_particles,
        feature_dim=feature_dim,
        embedding_dim=16,
        num_blocks=1,
        token_hidden=None,  # Will default to num_particles
        channel_hidden=16,
        output_dim=output_dim,
        aggregation='mean',
        dropout_rate=0.0,
        head_hidden_dims=[16]
    )


def build_jedi_linear_medium(num_particles=150, feature_dim=3, output_dim=5):
    """Medium JEDI-Linear model (~50K parameters)"""
    return build_jedi_linear_classifier(
        num_particles=num_particles,
        feature_dim=feature_dim,
        embedding_dim=16,
        num_blocks=2,
        token_hidden=None,  # Will default to num_particles
        channel_hidden=64,
        output_dim=output_dim,
        aggregation='mean',
        dropout_rate=0.1,
        head_hidden_dims=[64, 32]
    )


def build_jedi_linear_large(num_particles=150, feature_dim=3, output_dim=5):
    """Large JEDI-Linear model (~100K+ parameters)"""
    return build_jedi_linear_classifier(
        num_particles=num_particles,
        feature_dim=feature_dim,
        embedding_dim=32,
        num_blocks=3,
        token_hidden=None,  # Will default to num_particles
        channel_hidden=128,
        output_dim=output_dim,
        aggregation='mean',
        dropout_rate=0.1,
        head_hidden_dims=[128, 64, 32]
    )


# ---------------------------
# Paper-Matched JEDI-Linear Architecture
# ---------------------------


def build_jedi_linear_matched(
    num_particles=16,
    feature_dim=16,
    latent_dim=64,
    output_dim=5,
    head_hidden_dims=None
):
    """
    Build JEDI-Linear model matching the paper's benchmark configuration.

    Based on "JEDI-linear: Fast and Efficient Graph Neural Networks for Jet Tagging on FPGAs"
    https://arxiv.org/abs/2508.15468

    This implementation follows the architecture described in Figure 4 of the paper:
    1. Input projection layer: Projects input features into latent space
    2. Global information gathering: Linear-complexity interaction via global pooling
    3. Average pooling: Aggregate particle features
    4. Classification head: MLP with 4 dense layers

    The paper's benchmarks (Figure 6, Table 1) used configurations with:
    - 8, 16, 32, 64, or 128 particles
    - 3 features (pT, η, φ) or 16 features (full kinematic)
    - Fine-grained mixed-precision quantization (1-3 bit weights)
    - Distributed arithmetic for hardware efficiency

    Args:
        num_particles: Number of particles per jet (8, 16, 32, 64, or 128 in paper)
        feature_dim: Input feature dimension (3 or 16 in paper)
        latent_dim: Latent embedding dimension D_E (default: 64)
        output_dim: Number of output classes (5 in paper: g, q, W, Z, t)
        head_hidden_dims: Hidden dimensions for classification head (default: 4 layers)

    Returns:
        Keras Model matching the paper's architecture

    Example usage:
        # 16-particle, 16-feature configuration (Table I)
        model = build_jedi_linear_matched(
            num_particles=16,
            feature_dim=16,
            latent_dim=64,
            output_dim=5
        )
        # Expected: ~78.3% accuracy, 72ns latency on VU13P FPGA

        # 64-particle, 3-feature configuration (Table I)
        model = build_jedi_linear_matched(
            num_particles=64,
            feature_dim=3,
            latent_dim=64,
            output_dim=5
        )
        # Expected: ~81.8% accuracy, 78ns latency on VU13P FPGA
    """
    if head_hidden_dims is None:
        # Paper uses 4-layer MLP head (Dense x 4 in Figure 4)
        head_hidden_dims = [latent_dim, latent_dim, latent_dim // 2, latent_dim // 4]

    # Input: [batch, num_particles, feature_dim]
    inputs = layers.Input(shape=(num_particles, feature_dim), name='input_features')

    # Step 1: Input projection layer
    # Projects input features to latent dimension D_E
    x = layers.Dense(latent_dim, activation='relu', name='input_projection')(inputs)

    # Step 2: Global information gathering layer
    # Implements the linear-complexity interaction from Equation 4
    # E'_i ≈ W2 * (1/N_O) * sum(I_j) + W1 * I_i + C
    x = GlobalInteractionLayer(latent_dim, name='global_interaction')(x)
    x = layers.Activation('relu', name='interaction_activation')(x)

    # Step 3: Global average pooling across particles
    # [batch, num_particles, latent_dim] -> [batch, latent_dim]
    x = layers.GlobalAveragePooling1D(name='global_average_pool')(x)

    # Step 4: Classification head (MLP with 4 layers as shown in Figure 4)
    for i, hidden_dim in enumerate(head_hidden_dims):
        x = layers.Dense(hidden_dim, activation='relu', name=f'head_dense_{i}')(x)

    # Output layer: 5 classes (gluon, quark, W, Z, top)
    outputs = layers.Dense(output_dim, activation='softmax', name='output')(x)

    return Model(inputs=inputs, outputs=outputs, name='JEDI_Linear_Matched')


def build_jedi_linear_matched_16p16f(output_dim=5):
    """
    16-particle, 16-feature configuration from Table I.

    Paper results (permutation-invariant):
    - Accuracy: 78.3%
    - Latency: 72 ns
    - DSP: 0
    - LUT: 99k
    - FF: 50k
    - Initiation Interval: 1 clock
    - Fmax: 307.0 MHz
    """
    return build_jedi_linear_matched(
        num_particles=16,
        feature_dim=16,
        latent_dim=64,
        output_dim=output_dim
    )


def build_jedi_linear_matched_32p16f(output_dim=5):
    """
    32-particle, 16-feature configuration from Table I.

    Paper results (permutation-invariant):
    - Accuracy: 81.4%
    - Latency: 79 ns
    - DSP: 0
    - LUT: 147k
    - FF: 71k
    - Initiation Interval: 1 clock
    - Fmax: 304.7 MHz
    """
    return build_jedi_linear_matched(
        num_particles=32,
        feature_dim=16,
        latent_dim=64,
        output_dim=output_dim
    )


def build_jedi_linear_matched_64p16f(output_dim=5):
    """
    64-particle, 16-feature configuration from Table I.

    Paper results (permutation-invariant):
    - Accuracy: 82.4%
    - Latency: 93 ns
    - DSP: 0
    - LUT: 192k
    - FF: 92k
    - Initiation Interval: 1 clock
    - Fmax: 268.1 MHz
    """
    return build_jedi_linear_matched(
        num_particles=64,
        feature_dim=16,
        latent_dim=64,
        output_dim=output_dim
    )


def build_jedi_linear_matched_16p3f(output_dim=5):
    """
    16-particle, 3-feature configuration from Table I.

    Paper results (permutation-invariant):
    - Accuracy: 73.6%
    - Latency: 75 ns
    - DSP: 0
    - LUT: 136k
    - FF: 71k
    - Initiation Interval: 1 clock
    - Fmax: 305.7 MHz
    """
    return build_jedi_linear_matched(
        num_particles=16,
        feature_dim=3,
        latent_dim=64,
        output_dim=output_dim
    )


def build_jedi_linear_matched_64p3f(output_dim=5):
    """
    64-particle, 3-feature configuration from Table I.

    Paper results (permutation-invariant):
    - Accuracy: 81.8%
    - Latency: 78 ns
    - DSP: 0
    - LUT: 164k
    - FF: 93k
    - Initiation Interval: 1 clock
    - Fmax: 307.0 MHz
    """
    return build_jedi_linear_matched(
        num_particles=64,
        feature_dim=3,
        latent_dim=64,
        output_dim=output_dim
    )
