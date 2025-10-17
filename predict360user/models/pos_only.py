import logging
import os
from contextlib import suppress
from typing import Sequence, Tuple

import absl
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense, Input, Lambda
from sklearn.utils.validation import check_is_fitted
from tensorflow import keras
from wandb.keras import WandbCallback
from tensorflow.keras.optimizers.legacy import Adam  # ✅ Faster on M1/M2 GPUs

import wandb
from predict360user.base_model import BaseModel, batch_generator_fn
from predict360user.run_config import RunConfig
from predict360user.utils.math360 import (
    cartesian_to_eulerian,
    eulerian_to_cartesian,
    metric_orth_dist_eulerian,
)

log = logging.getLogger()


# --- Delta Angle Helper ------------------------------------------------------
def delta_angle_from_ori_mag_dir(values):
    orientation = values[0]
    magnitudes = values[1] / 2.0
    directions = values[2]
    motion = magnitudes * directions

    yaw_pred_wo_corr = orientation[:, :, 0:1] + motion[:, :, 0:1]
    pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]

    cond_above = tf.cast(tf.greater(pitch_pred_wo_corr, 1.0), tf.float32)
    cond_correct = tf.cast(
        tf.logical_and(
            tf.less_equal(pitch_pred_wo_corr, 1.0),
            tf.greater_equal(pitch_pred_wo_corr, 0.0),
        ),
        tf.float32,
    )
    cond_below = tf.cast(tf.less(pitch_pred_wo_corr, 0.0), tf.float32)

    pitch_pred = (
        cond_above * (1.0 - (pitch_pred_wo_corr - 1.0))
        + cond_correct * pitch_pred_wo_corr
        + cond_below * (-pitch_pred_wo_corr)
    )
    yaw_pred = tf.math.mod(
        cond_above * (yaw_pred_wo_corr - 0.5)
        + cond_correct * yaw_pred_wo_corr
        + cond_below * (yaw_pred_wo_corr - 0.5),
        1.0,
    )
    return tf.concat([yaw_pred, pitch_pred], -1)


# --- Coordinate Conversion Utilities ----------------------------------------
def batch_cartesian_to_normalized_eulerian(positions_in_batch: np.ndarray) -> np.ndarray:
    eulerian_batch = [
        [cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch]
        for batch in positions_in_batch
    ]
    eulerian_batch = np.array(eulerian_batch) / np.array([2 * np.pi, np.pi])
    return eulerian_batch


def batch_normalized_eulerian_to_cartesian(positions_in_batch: np.ndarray) -> np.ndarray:
    positions_in_batch = positions_in_batch * np.array([2 * np.pi, np.pi])
    cartesian_batch = [
        [eulerian_to_cartesian(pos[0], pos[1]) for pos in batch]
        for batch in positions_in_batch
    ]
    return cartesian_batch


# --- Model Definition --------------------------------------------------------
class PosOnly(BaseModel):
    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg

    def get_model(self) -> keras.Model:
        encoder_inputs = Input(shape=(self.cfg.m_window, 2))
        decoder_inputs = Input(shape=(1, 2))

        lstm_layer = LSTM(256, return_sequences=True, return_state=True)
        decoder_dense_mot = Dense(2, activation="sigmoid")
        decoder_dense_dir = Dense(2, activation="tanh")
        To_Position = Lambda(delta_angle_from_ori_mag_dir)

        # Encode → Decode loop
        _, state_h, state_c = lstm_layer(encoder_inputs)
        states = [state_h, state_c]

        all_outputs = []
        inputs = decoder_inputs
        for _ in range(self.cfg.h_window):
            decoder_pred, state_h, state_c = lstm_layer(inputs, initial_state=states)
            outputs_delta = decoder_dense_mot(decoder_pred)
            outputs_delta_dir = decoder_dense_dir(decoder_pred)
            outputs_pos = To_Position([inputs, outputs_delta, outputs_delta_dir])
            all_outputs.append(outputs_pos)
            inputs = outputs_pos
            states = [state_h, state_c]

        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
        model = keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
        model_optimizer = Adam(learning_rate=self.cfg.lr)
        model.compile(optimizer=model_optimizer, loss=metric_orth_dist_eulerian)
        return model

    # --- Training ------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> BaseModel:
        log.info("fit ...")

        absl.logging.set_verbosity(absl.logging.ERROR)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        self.model = self.get_model()
        if wandb.run.resumed:
            try:
                log.info("restoring fit from previous interrupted.")
                self.model.load_weights(wandb.restore("model-best.h5").name)
                self.cfg.initial_epoch = wandb.run.step
            except:
                log.error("restoring fit failed. starting new fit.")

        if self.cfg.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cfg.gpu_id)
            log.info(f"set visible gpu to {self.cfg.gpu_id}")

        train_wins = df[df["partition"] == "train"]
        val_wins = df[df["partition"] == "val"]
        steps_per_ep_train = np.ceil(len(train_wins) / self.cfg.batch_size)
        steps_per_ep_validate = np.ceil(len(val_wins) / self.cfg.batch_size)

        def get_fit_data(df: pd.DataFrame) -> Tuple[list, list]:
            encoder_pos_inputs = df["m_window"].values
            decoder_pos_inputs = df["trace"].values
            decoder_outputs = df["h_window"].values
            return (
                [
                    batch_cartesian_to_normalized_eulerian(encoder_pos_inputs),
                    batch_cartesian_to_normalized_eulerian(decoder_pos_inputs),
                ],
                batch_cartesian_to_normalized_eulerian(decoder_outputs),
            )

        self.model.fit(
            batch_generator_fn(self.cfg.batch_size, train_wins, get_fit_data),
            validation_data=batch_generator_fn(self.cfg.batch_size, val_wins, get_fit_data),
            steps_per_epoch=steps_per_ep_train,
            validation_steps=steps_per_ep_validate,
            epochs=self.cfg.epochs,
            initial_epoch=self.cfg.initial_epoch,
            callbacks=[WandbCallback(save_model=True, monitor="loss")],
            verbose=2,
        )
        self.is_fitted_ = True
        return self

    # --- Prediction ----------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> Sequence:
        log.info("predict ...")
        check_is_fitted(self)
        encoder_pos_inputs = df["m_window"].values
        decoder_pos_inputs = df["trace"].values
        predict_data = [
            batch_cartesian_to_normalized_eulerian(encoder_pos_inputs),
            batch_cartesian_to_normalized_eulerian(decoder_pos_inputs),
        ]
        pred = self.model.predict(predict_data, verbose=2)
        return batch_normalized_eulerian_to_cartesian(pred)

    # --- Evaluation ----------------------------------------------------------
    def evaluate(self, df: pd.DataFrame):
        log.info("evaluate ...")
        check_is_fitted(self)

        test_wins = df[df["partition"] == "test"]
        encoder_pos_inputs = test_wins["m_window"].values
        decoder_pos_inputs = test_wins["trace"].values
        true_outputs = test_wins["h_window"].values

        predict_data = [
            batch_cartesian_to_normalized_eulerian(encoder_pos_inputs),
            batch_cartesian_to_normalized_eulerian(decoder_pos_inputs),
        ]

        pred_outputs = self.model.predict(predict_data, verbose=2)
        pred_cartesian = np.array(batch_normalized_eulerian_to_cartesian(pred_outputs))
        true_cartesian = np.array(true_outputs)

        # Mean Angular Error
        distances = np.array([
            np.mean([
                np.arccos(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1.0, 1.0))
                for a, b in zip(true_seq, pred_seq)
            ])
            for true_seq, pred_seq in zip(true_cartesian, pred_cartesian)
        ])
        mean_err = np.degrees(np.mean(distances))
        wandb.log({"test_mean_angular_error_deg": mean_err})
        print(f"Mean Angular Error (degrees): {mean_err:.2f}")

        # --- Visualizations ---
        flat_path, globe_true, globe_pred = spherical_heatmap(true_cartesian, pred_cartesian)
        try:
            wandb.log({
                "spherical_heatmap_flat": wandb.Image(flat_path),
                "spherical_globe_true": wandb.Image(globe_true),
                "spherical_globe_pred": wandb.Image(globe_pred),
            })
        except Exception as e:
            print(f"⚠️ Could not log heatmaps to W&B: {e}")

        return {
            "test_mean_angular_error_deg": mean_err,
            "y_true": true_cartesian,
            "y_pred": pred_cartesian
        }


# --- Heatmap Visualization (2D + 3D Globe) ----------------------------------
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def spherical_heatmap(y_true, y_pred, save_prefix="spherical_heatmap"):
    """Generate equirectangular + 3D globe heatmaps."""
    def to_equirectangular(coords):
        yaw = np.arctan2(coords[:, 1], coords[:, 0])
        pitch = np.arcsin(coords[:, 2] / np.linalg.norm(coords, axis=1))
        return np.degrees(yaw), np.degrees(pitch)

    y_true_flat = np.reshape(y_true, (-1, 3))
    y_pred_flat = np.reshape(y_pred, (-1, 3))
    yaw_t, pitch_t = to_equirectangular(y_true_flat)
    yaw_p, pitch_p = to_equirectangular(y_pred_flat)

    kde_true = gaussian_kde([yaw_t, pitch_t])
    kde_pred = gaussian_kde([yaw_p, pitch_p])
    yaw_grid, pitch_grid = np.meshgrid(np.linspace(-180, 180, 300), np.linspace(-90, 90, 150))
    true_density = kde_true([yaw_grid.ravel(), pitch_grid.ravel()]).reshape(yaw_grid.shape)
    pred_density = kde_pred([yaw_grid.ravel(), pitch_grid.ravel()]).reshape(yaw_grid.shape)

    # Flat 2D heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(true_density, extent=(-180, 180, -90, 90), cmap='viridis', origin='lower')
    axes[0].set_title("True Gaze Distribution")
    axes[1].imshow(pred_density, extent=(-180, 180, -90, 90), cmap='plasma', origin='lower')
    axes[1].set_title("Predicted Gaze Distribution")
    for ax in axes:
        ax.set_xlabel("Yaw (°)")
        ax.set_ylabel("Pitch (°)")
    flat_path = f"{save_prefix}_flat.png"
    plt.savefig(flat_path, dpi=200)
    plt.close()

    # 3D sphere projection
    def plot_spherical(density, title, save_path, cmap):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        yaw_rad = np.radians(yaw_grid)
        pitch_rad = np.radians(pitch_grid)
        x = np.cos(pitch_rad) * np.cos(yaw_rad)
        y = np.cos(pitch_rad) * np.sin(yaw_rad)
        z = np.sin(pitch_rad)
        norm_density = (density - density.min()) / (density.max() - density.min())
        ax.plot_surface(
            x, y, z,
            facecolors=plt.cm.get_cmap(cmap)(norm_density),
            rstride=2, cstride=2, linewidth=0, antialiased=False, shade=False
        )
        ax.set_title(title)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=120)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    globe_true = f"{save_prefix}_globe_true.png"
    globe_pred = f"{save_prefix}_globe_pred.png"
    plot_spherical(true_density, "True Gaze (3D Sphere)", globe_true, "viridis")
    plot_spherical(pred_density, "Predicted Gaze (3D Sphere)", globe_pred, "plasma")

    print(f"🌍 Saved equirectangular heatmap: {flat_path}")
    print(f"🪐 Saved 3D true gaze globe: {globe_true}")
    print(f"🪐 Saved 3D predicted gaze globe: {globe_pred}")

    return flat_path, globe_true, globe_pred
