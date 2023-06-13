# How to configure experiments
Our experiments are managed through [Hydra configs](https://github.com/facebookresearch/hydra).

The hydra configuration hierarchy looks like:
```shell
configs/
    config.yaml          # the default config file that summarizes all configs
    data/default.yaml    # the configs related to data
    eval/default.yaml    # the configs related to evaluation
    lifelong
        base.yaml        # the sequential finetuning baseline configs
        agem.yaml        # the agem configs
        er.yaml          # the er configs
        ewc.yaml         # the ewc configs
        packnet.yaml     # the packnet configs
        multitask.yaml   # the multitask learning configs
        single_task.yaml # the single task learning configs
    policy
        data_augmentation
        image_encoder
        language_encoder
        policy_head
        position_encoding
        bc_rnn_policy.yaml          # the config for ResNet-LSTM
        bc_transformer_policy.yaml  # the config for ResNet-Transformer
        bc_vilt_policy.yaml         # the config for ViT-Transformer
    train
        optimizer
        shceduler
        default.yaml      # the configs related to training
```

If you want to modify any existing configuration, you can directly do that in command line instead of modifying
the original yaml file. For instance, consider
```
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python lifelong/main.py seed=SEED \
                        benchmark_name=BENCHMARK 
                        policy=POLICY \
                        lifelong=ewc \
                        lifelong.e_lambda=100000 \
                        lifelong.gamma=0.95
```
This will change the ```e_lambda``` and ```gamma``` configs of ewc to ```100000``` and ```0.95``` respectively.
