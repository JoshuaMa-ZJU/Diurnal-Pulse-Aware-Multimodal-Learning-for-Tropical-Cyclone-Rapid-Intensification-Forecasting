# Dataset

The large data files for this project are not stored in GitHub. They are hosted on IEEE DataPort.

**IEEE DataPort dataset:** TODO: add IEEE DataPort DOI or URL

Place the downloaded files in this directory as follows:

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

## File Description

- `x.npy`: shape `(13336, 4, 256, 256, 9)`. Channel 0 is IR satellite imagery, channels 1-6 are GPH fields, channel 7 is SST, and channel 8 is SSS.
- `x_2d.npy`: shape `(13336, 4, 6)`. Historical 2D predictors, including MSW, latitude, longitude, year, month, and day/time information.
- `dp.npy`: shape `(13336, 4, 256, 256)`. Diurnal-pulse feature maps.
- `wind_y.npy`: shape `(13336, 4)`. Ground-truth future MSW labels.
- `prediction_result.npy` and `ground_truth.npy`: saved prediction and label arrays for the overall test set.
- `prediction_result_ri.npy` and `ground_truth_ri.npy`: saved prediction and label arrays for the RI subset.

The dataset contains 13,336 samples from January 2019 to September 2024, including 3,641 RI samples.
