"""Arquitecturas del clasificador de estrés biométrico (TensorFlow/Keras).

Regla fija del proyecto: TensorFlow/Keras en todo el backend, no PyTorch.
Se implementan y comparan dos arquitecturas bajo una interfaz homogénea
(mismo formato de hiperparámetros de entrada -> `tf.keras.Model` compilado),
para que la comparación de la Fase 7 sea justa y no favorezca a ninguna por
diferencias de interfaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class CNNLSTMHyperparams:
    """Hiperparámetros de la arquitectura CNN-1D + LSTM (sobre ventana cruda)."""

    conv_filters: tuple[int, ...] = (32, 64)
    conv_kernel_size: int = 5
    lstm_units: int = 64
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    dense_units: int = 32

    def to_dict(self) -> dict[str, Any]:
        return {
            "conv_filters": list(self.conv_filters),
            "conv_kernel_size": self.conv_kernel_size,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "dense_units": self.dense_units,
        }


@dataclass
class FeaturesMLPHyperparams:
    """Hiperparámetros de la arquitectura Features (HRV/EDA) + MLP."""

    hidden_layers: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    l2_regularization: float = 1e-4

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_layers": list(self.hidden_layers),
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "l2_regularization": self.l2_regularization,
        }


def build_cnn_lstm_model(
    input_shape: tuple[int, int], n_classes: int, hp: CNNLSTMHyperparams | None = None
) -> tf.keras.Model:
    """CNN-1D + LSTM sobre ventanas de señal cruda multicanal.

    input_shape = (longitud_ventana, n_canales).
    """
    hp = hp or CNNLSTMHyperparams()

    inputs = layers.Input(shape=input_shape, name="raw_signal_window")
    x = inputs
    for n_filters in hp.conv_filters:
        x = layers.Conv1D(n_filters, hp.conv_kernel_size, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(hp.dropout_rate)(x)

    x = layers.LSTM(hp.lstm_units)(x)
    x = layers.Dense(hp.dense_units, activation="relu")(x)
    x = layers.Dropout(hp.dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="stress_class")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_lstm_stress_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_features_mlp_model(
    input_dim: int, n_classes: int, hp: FeaturesMLPHyperparams | None = None
) -> tf.keras.Model:
    """MLP sobre el vector de features HRV/EDA extraído por ventana."""
    hp = hp or FeaturesMLPHyperparams()

    inputs = layers.Input(shape=(input_dim,), name="hrv_eda_features")
    x = inputs
    for n_units in hp.hidden_layers:
        x = layers.Dense(
            n_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(hp.l2_regularization),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="stress_class")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="features_mlp_stress_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


ARCHITECTURE_BUILDERS = {
    "cnn_lstm": build_cnn_lstm_model,
    "features_mlp": build_features_mlp_model,
}
