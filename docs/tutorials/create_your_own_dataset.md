# Create Your Own Datasets


## Collect demonstrations

A spacemouse is required for launching the data collection script. Replace `PDDL_FILE_NAME` with an actual file path.

```shell
    python scripts/collect_demonstration.py \
                       --controller OSC_POSE \
                       --camera agentview --robots Panda \
                       --num-demonstration 50 \
                       --rot-sensitivity 1.5 \
                       --bddl-file PDDL_FILE_NAME
```

## Create hdf5 dataset files

If you are creating your own datasets, it is likely to have demonstrations collected for multiple tasks. To create all the demonstration data in a batch, we provide the following script:

```shell
    python scripts/batch_create_dataset.py --folder-path PATH_TO_FILES
```

You would need to put all the folders created from previous step inside the path `PATH_TO_FILES`. For more details of creating a single hdf5 file for demonstrations, we refer to the file [`scripts/create_dataset.py`]().
