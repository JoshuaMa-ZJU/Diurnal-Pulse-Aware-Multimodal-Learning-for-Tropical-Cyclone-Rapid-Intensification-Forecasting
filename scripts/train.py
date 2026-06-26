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
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-fraction", type=float, default=1 / 6)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    x = np.load(args.data_dir / "x.npy", mmap_mode="r")
    x_2d = np.load(args.data_dir / "x_2d.npy", mmap_mode="r")
    dp = np.load(args.data_dir / "dp.npy", mmap_mode="r")
    y = np.load(args.data_dir / "wind_y.npy", mmap_mode="r")

    n_total = len(y)
    n_test = batch_aligned_length(int(n_total * args.test_fraction), args.batch_size)
    n_train = batch_aligned_length(n_total - n_test, args.batch_size)

    train_slice = slice(0, n_train)
    decoder_inputs = np.asarray(x_2d[train_slice, :, 0]).reshape(n_train, 4, 1)

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
            np.asarray(x[train_slice]).astype(np.float16),
            np.asarray(x_2d[train_slice]).astype(np.float16),
            np.asarray(dp[train_slice]).astype(np.float16),
            decoder_inputs.astype(np.float16),
        ],
        np.asarray(y[train_slice]).astype(np.float16),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        validation_split=args.validation_split,
        shuffle=True,
        verbose=1,
    )

    model.load_weights(args.output_dir / "model.weights.h5")
    test_slice = slice(n_total - n_test, n_total)
    test_x_2d = np.asarray(x_2d[test_slice]).astype(np.float16)
    test_decoder_inputs = test_x_2d[:, :, 0].reshape(n_test, 4, 1)
    predictions = model.predict(
        [
            np.asarray(x[test_slice]).astype(np.float16),
            test_x_2d,
            np.asarray(dp[test_slice]).astype(np.float16),
            test_decoder_inputs.astype(np.float16),
        ],
        batch_size=4,
    )
    np.save(args.output_dir / "prediction_result.npy", predictions.squeeze(-1))
    np.save(args.output_dir / "ground_truth.npy", np.asarray(y[test_slice]))


if __name__ == "__main__":
    main()
