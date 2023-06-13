# Datasets
We provide 4 datasets for the study in lifelong learning in desicion making (LLDM): `libero_spatial`, `libero_object`, `libero_goal` and `libero_100`, which consists of 10, 10, 10, and 100 tasks each.

To download all datasets:
```shell
   python benchmark_scripts/download_libero_datasets.py
```
All data will be downloaded in the directory `./datasets`.

If you only want to download a specific dataset, use (take `libero_100` as an example):
```shell
   python benchmark_scripts/download_libero_datasets.py --datasets libero_100
```
