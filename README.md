Support training the classification network on histology datasets.

## Code Organizations

Run [`main.py`](main.py) to train the models.

Tune all the hyper-parameters in [`config.yaml`](config.yaml)
- `train_root`: Path to the training set.
- `test_root`: Path to the test set.
- `output_path`: Path to the output. Output files will be exported to a folder created in `output_path` started with the date, hence no worry for overriding.

## Dataset

Datasets can be downloadee use [`preprocess/download.py`](preprocess/download.py).

Normalized-v1: Template `NORM-AAAWMSFI.tif` (from training set).
Normalized-v2: Template `STR-AAEILWWE.tif` (from training set).
Normalized-v3: Template `NORM-TCGA-AASSYQPA.tif` (from test set).

## Results
| *ResNet-18* | w/o Pretrain | w/ Pretrain |
| -- | -- | -- |
| w/o Norma | 64.958 | 58.788 |
| w/ Norm v1 | 78.914 | |
| w/ Norm v2 | | |
| w/ Norm v3 | | |

