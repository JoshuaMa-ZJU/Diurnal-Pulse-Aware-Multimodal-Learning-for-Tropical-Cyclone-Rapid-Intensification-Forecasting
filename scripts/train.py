import argparse
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import build_model


def batch_aligned_length(n_samples, batch_size):
    return (n_samples // batch_size) * batch_size


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TC intensity forecasting model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing event-level train_indices.npy, val_indices.npy, "
            "and test_indices.npy. If omitted, a 4:1:1 sequential split is used."
        ),
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def load_split_indices(args, n_total):
    if args.split_dir is not None:
        return (
            np.load(args.split_dir / "train_indices.npy"),
            np.load(args.split_dir / "val_indices.npy"),
            np.load(args.split_dir / "test_indices.npy"),
        )

    n_unit = n_total // 6
    train_end = n_unit * 4
    val_end = n_unit * 5
    return (
        np.arange(0, train_end),
        np.arange(train_end, val_end),
        np.arange(val_end, n_total),
    )


def align_indices(indices, batch_size):
    n = batch_aligned_length(len(indices), batch_size)
    return indices[:n]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    x = np.load(args.data_dir / "x.npy", mmap_mode="r")
    x_2d = np.load(args.data_dir / "x_2d.npy", mmap_mode="r")
    dp = np.load(args.data_dir / "dp.npy", mmap_mode="r")
    y = np.load(args.data_dir / "wind_y.npy", mmap_mode="r")

    train_indices, val_indices, test_indices = load_split_indices(args, len(y))
    train_indices = align_indices(train_indices, args.batch_size)
    val_indices = align_indices(val_indices, args.batch_size)
    test_indices = align_indices(test_indices, args.batch_size)

    decoder_inputs = np.asarray(x_2d[train_indices, :, 0]).reshape(len(train_indices), 4, 1)
    val_decoder_inputs = np.asarray(x_2d[val_indices, :, 0]).reshape(len(val_indices), 4, 1)

    model = build_model()
    model.compile(optimizer="adam", loss="mae", metrics=["mae", "mse"])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            args.output_dir / "model.weights.h5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=20,
            verbose=1,
            min_delta=1e-30,
            min_lr=1e-30,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-5,
            patience=80,
            verbose=1,
        ),
    ]

    model.fit(
        [
            np.asarray(x[train_indices]).astype(np.float16),
            np.asarray(x_2d[train_indices]).astype(np.float16),
            np.asarray(dp[train_indices]).astype(np.float16),
            decoder_inputs.astype(np.float16),
        ],
        np.asarray(y[train_indices]).astype(np.float16),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_data=(
            [
                np.asarray(x[val_indices]).astype(np.float16),
                np.asarray(x_2d[val_indices]).astype(np.float16),
                np.asarray(dp[val_indices]).astype(np.float16),
                val_decoder_inputs.astype(np.float16),
            ],
            np.asarray(y[val_indices]).astype(np.float16),
        ),
        shuffle=True,
        verbose=1,
    )

    model.load_weights(args.output_dir / "model.weights.h5")
    test_x_2d = np.asarray(x_2d[test_indices]).astype(np.float16)
    test_decoder_inputs = test_x_2d[:, :, 0].reshape(len(test_indices), 4, 1)
    predictions = model.predict(
        [
            np.asarray(x[test_indices]).astype(np.float16),
            test_x_2d,
            np.asarray(dp[test_indices]).astype(np.float16),
            test_decoder_inputs.astype(np.float16),
        ],
        batch_size=4,
    )
    np.save(args.output_dir / "prediction_result.npy", predictions.squeeze(-1))
    np.save(args.output_dir / "ground_truth.npy", np.asarray(y[test_indices]))


if __name__ == "__main__":
    main()
