# Code Structures

Here is the overview of the codebase with some comments.

## Enviornments
```shell
libero/
    assets/ # Where all the object mesh files are stored
    bddl_files/   # PDDL file definitions for tasks
        libero_goal/*.bddl      # 10 tasks of LIBERO-Goal suite
        libero_spatial/*.bddl   # 10 tasks of LIBERO-Spatial suite
        libero_object/*.bddl    # 10 tasks of LIBERO-Object suite
        libero_10/*.bddl        # 10 tasks of LIBERO-100 for evaluation (aka LIBERO-LONG)
        libero_90/*.bddl        # 90 tasks of LIBERO-100 for pretraining

    benchmark/    # Task orders for evaluation of all benchmarks
    envs/         # Environment definitions for LIBERO tasks
    init_files/   # Fixed initializations for benchmark evaluation
    utils/        # Miscellaneous utility functions
```

## Policy, algorithms, and experiments
```shell
lifelong/      # Files for algorithms, models, and training / testing
    main.py    # the main script for reproducing experiments.
    algos/
        base.py                 # Base class `Sequential` for all the algorithms
        er.py                   # Algorithm Experience Replay
        ewc.py                  # Algorithm Elastic Weight Consolidation
        packnet.py              # Algorithm Packnet
        multitask.py            # Algorithm multitask (baseline)
        single_task.py          # Algorithm single task (baseline)
        language.py

    models/
        policy/
            bc_rnn_policy.py            # ResNet-RNN
            bc_transformer_policy.py    # ResNet-T
            bc_vilt_policy.py           # ViT-T
```


## Scripts for dataset creation
```shell
scripts/
    collect_demonstrations.py    # Collect your own demonstrations
    create_dataset.py            # Create your own dataset
    batch_create_dataset.py      # Create a batch of datasets by calling create_dataset.py repeatedly
```
