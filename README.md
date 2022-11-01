Support training the classification network on histology datasets.

Run [`main.py`](main.py) to train the models.

Tune all the hyper-parameters in [`config.yaml`](config.yaml):
    - `train_root`: Path to the training set.
    - `test_root`: Path to the test set.
    - `output_path`: Path to the output. Output files will be exported to a folder created in `output_path` started with the date, hence no worry for overriding.

Datasets can be downloadee use [`download.py`](download.py).
