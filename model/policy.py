"""Multimodal temporal driving policy.

Per timestep: RGB CNN + LiDAR-BEV CNN + waypoint MLP + telemetry MLP, fused
and projected to 256-d. A positional embedding and two Transformer blocks mix
the temporal stack; the last timestep predicts [throttle, brake, steer].

Note: the positional embedding below is applied to a concrete tf.range, so
Keras 3 evaluates it once at build time. It is a fixed (randomly initialised,
non-trainable) constant in the graph rather than a learned embedding. It is
kept as-is to match the architecture of the released policy.
"""
import tensorflow as tf


def build_policy(stack_size, vision_size, lidar_shape, telemetry_shape, waypoints_shape):
    v_in = tf.keras.Input(shape=(stack_size,) + tuple(vision_size) + (3,), name="vision")
    l_in = tf.keras.Input(shape=(stack_size,) + tuple(lidar_shape), name="lidar")
    t_in = tf.keras.Input(shape=(stack_size,) + tuple(telemetry_shape), name="telemetry")
    w_in = tf.keras.Input(shape=(stack_size,) + tuple(waypoints_shape), name="waypoints")

    # Vision encoder (per timestep)
    v = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
    )(v_in)
    v = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")
    )(v)
    v = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")
    )(v)
    v = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(v)  # (B, T, 128)

    # LiDAR encoder (per timestep)
    l = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
    )(l_in)
    l = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")
    )(l)
    l = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")
    )(l)
    l = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(l)  # (B, T, 128)

    # Waypoints (per timestep)
    w = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(w_in)
    w = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu"))(w)  # (B, T, 32)

    # Telemetry (per timestep)
    t = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(t_in)
    t = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation="relu"))(t)  # (B, T, 32)

    # Fusion (per timestep)
    x = tf.keras.layers.Concatenate()([v, l, w, t])  # (B, T, 320)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    # Positional embedding (baked in as a constant at build time, see module docstring)
    pos_emb = tf.keras.layers.Embedding(input_dim=stack_size, output_dim=256)
    x = x + pos_emb(tf.range(start=0, limit=stack_size, delta=1))

    def transformer_block(x):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = tf.keras.layers.Add()([x, attn])
        x = tf.keras.layers.LayerNormalization()(x)
        ffn = tf.keras.layers.Dense(512, activation="relu")(x)
        ffn = tf.keras.layers.Dense(256)(ffn)
        x = tf.keras.layers.Add()([x, ffn])
        return tf.keras.layers.LayerNormalization()(x)

    x = transformer_block(x)
    x = transformer_block(x)

    # Last timestep -> control head
    x = x[:, -1, :]
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(3, name="controls")(x)

    return tf.keras.Model(inputs=[v_in, l_in, t_in, w_in], outputs=out)


def load_policy(path):
    return tf.keras.models.load_model(str(path), safe_mode=False, compile=False)
