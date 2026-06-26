import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error


LEAD_TIMES = [6, 12, 18, 24]


def mae(pred, label):
    return np.mean(np.abs(pred - label), axis=0)


def rmse(pred, label):
    return np.array(
        [mean_squared_error(label[:, i], pred[:, i], squared=False) for i in range(label.shape[1])]
    )


def report(name, pred_path, label_path):
    pred = np.load(pred_path)
    label = np.load(label_path)
    if pred.ndim == 3 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    print(f"\n{name}")
    print("-" * len(name))
    mae_values = mae(pred, label)
    rmse_values = rmse(pred, label)
    for idx, lead_time in enumerate(LEAD_TIMES):
        print(
            f"{lead_time:>2}h: MAE = {mae_values[idx]:.3f} kt, "
            f"RMSE = {rmse_values[idx]:.3f} kt"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TC intensity prediction arrays.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main():
    args = parse_args()
    report(
        "Overall test set",
        args.data_dir / "prediction_result.npy",
        args.data_dir / "ground_truth.npy",
    )

    ri_pred = args.data_dir / "prediction_result_ri.npy"
    ri_label = args.data_dir / "ground_truth_ri.npy"
    if ri_pred.exists() and ri_label.exists():
        report("RI subset", ri_pred, ri_label)


if __name__ == "__main__":
    main()
