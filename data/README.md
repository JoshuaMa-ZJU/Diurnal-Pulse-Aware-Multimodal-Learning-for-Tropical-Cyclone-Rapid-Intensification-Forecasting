# Dataset

Small data files and saved prediction arrays are included in `data/small/`. Large files are hosted on Google Drive.

Large files:

- `dp.zip`: https://drive.google.com/file/d/1fWDIZUbZn1Bt3tyOCKruZAfaidn97IAy/view?usp=drive_link
- `x.rar`: https://drive.google.com/file/d/1fc1DzjvZxJmQepbMvUnk6lcRXUlWlG37/view?usp=drive_link

For full training, download and extract the large files, then place all required files in `data/raw/`:

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

Small files included in this repository:

```text
data/small/
|-- x_2d.npy
|-- wind_y.npy
|-- prediction_result.npy
|-- ground_truth.npy
|-- prediction_result_ri.npy
|-- ground_truth_ri.npy
`-- readme.txt
```

## File Description

- `x.npy`: shape `(13336, 4, 256, 256, 9)`. Channel 0 is IR satellite imagery, channels 1-6 are GPH fields, channel 7 is SST, and channel 8 is SSS.
- `x_2d.npy`: shape `(13336, 4, 6)`. Historical 2D predictors, including MSW, latitude, longitude, year, month, and day/time information.
- `dp.npy`: shape `(13336, 4, 256, 256)`. Diurnal-pulse feature maps.
- `wind_y.npy`: shape `(13336, 4)`. Ground-truth future MSW labels.
- `prediction_result.npy` and `ground_truth.npy`: saved prediction and label arrays for the overall test set.
- `prediction_result_ri.npy` and `ground_truth_ri.npy`: saved prediction and label arrays for the RI subset.

The dataset contains 13,336 samples from January 2019 to September 2024, including 3,641 RI samples.
