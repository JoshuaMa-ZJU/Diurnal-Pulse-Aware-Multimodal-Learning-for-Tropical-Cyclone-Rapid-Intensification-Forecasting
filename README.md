# Diurnal Pulse-Aware Multimodal Learning for Tropical Cyclone Rapid Intensification Forecasting

This repository provides code and documentation for the paper:

**Diurnal Pulse-Aware Multimodal Learning with Cross-Attention and Masked Convolution for Tropical Cyclone Rapid Intensification Forecasting**

The proposed model forecasts tropical cyclone (TC) intensity at 6-, 12-, 18-, and 24-hour lead times, with special emphasis on rapid intensification (RI). It integrates infrared satellite imagery, geopotential height fields, sea surface temperature, sea surface salinity, diurnal-pulse features, historical intensity, location, and temporal information.

## Highlights

- RI-oriented TC intensity forecasting at four 6-hourly lead times.
- Diurnal-pulse-aware RI representation from cold cloud-top evolution.
- Symmetric cross-attention for coupled SST and SSS feature extraction.
- Circular and directional masked convolution for storm-structure representation.
- Evaluation on a global TC dataset from 2019 to 2024.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- LICENSE
|-- CITATION.cff
|-- data/
|   `-- README.md
|-- src/
|   |-- __init__.py
|   `-- model.py
`-- scripts/
    |-- train.py
    `-- evaluate.py
```

## Dataset

Large data files are hosted on IEEE DataPort rather than stored in this GitHub repository.

**IEEE DataPort dataset:** https://ieee-dataport.org/documents/diurnal-pulse-aware-image-feature-extraction

After downloading the dataset, place the files under `data/raw/`:

```text
data/raw/
|-- x.npy
|-- x_2d.npy
|-- dp.npy
|-- wind_y.npy
|-- prediction_result.npy
|-- ground_truth.npy
|-- prediction_result_ri.npy
`-- ground_truth_ri.npy
```

The processed dataset contains 13,336 samples from January 2019 to September 2024, including 3,641 RI samples. TC cases cover six basins: Western Pacific, Eastern Pacific, Southern Pacific, North Indian, South Indian, and North Atlantic.

## Main Data Files

| File | Shape | Description |
| --- | --- | --- |
| `x.npy` | `(13336, 4, 256, 256, 9)` | Multimodal gridded inputs. Channel 0 is IR satellite imagery, channels 1-6 are GPH fields, channel 7 is SST, and channel 8 is SSS. |
| `x_2d.npy` | `(13336, 4, 6)` | Historical non-image predictors, including MSW, latitude, longitude, year, month, and day/time information. |
| `dp.npy` | `(13336, 4, 256, 256)` | Diurnal-pulse feature maps. |
| `wind_y.npy` | `(13336, 4)` | Future MSW labels at 6-, 12-, 18-, and 24-hour lead times. |

## Data Sources

The processed dataset is derived from:

- IBTrACS for TC track, maximum sustained wind, location, classification, and temporal information.
- GridSat-B1 for global geostationary satellite imagery. The IR channel is used in this study.
- ECMWF HRES for short-term geopotential height forecasts.
- HYCOM GOFS 3.1 for sea surface temperature and sea surface salinity.

## Installation

```bash
pip install -r requirements.txt
```

The implementation uses TensorFlow/Keras with mixed precision. Experiments in the paper were trained on a single NVIDIA RTX 4090 GPU.

## Training

```bash
python scripts/train.py --data-dir data/raw --output-dir outputs --epochs 800 --batch-size 16
```

Default training settings:

- optimizer: Adam
- loss: MAE
- batch size: 16
- learning-rate reduction: factor 0.5 after 20 stagnant validation epochs
- early stopping: 80 stagnant validation epochs
- mixed precision: enabled

## Evaluation

To evaluate saved prediction arrays:

```bash
python scripts/evaluate.py --data-dir data/raw
```

The script reports MAE and RMSE at 6-, 12-, 18-, and 24-hour lead times for the full test set and, when available, the RI subset.

## Model Overview

The model contains four main components:

1. **Cloud feature extraction module**: extracts multi-scale convective structures from IR imagery using circular masks.
2. **Atmospheric feature extraction module**: extracts storm-structure information from GPH fields using directional masks.
3. **Oceanic feature extraction module**: models SST-SSS interactions using symmetric cross-attention.
4. **Diurnal-pulse and RI detection module**: injects DP representations into the Transformer encoder-decoder to improve responsiveness to RI signals.

## Citation

```bibtex
@article{ma2026diurnal,
  title   = {Diurnal Pulse-Aware Multimodal Learning with Cross-Attention and Masked Convolution for Tropical Cyclone Rapid Intensification Forecasting},
  author  = {Ma, Zhaoyang and Tang, Huan and Lin, Jianmin and Ma, Dongfang},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Under review}
}
```

## License

The code is released under the MIT License. The dataset is distributed separately through IEEE DataPort and should be used according to the license and terms provided there.

## Contact

For questions, please contact the corresponding author listed in the paper.
