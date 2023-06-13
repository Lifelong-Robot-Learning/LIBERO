# Pretraining effects
We use ```libero_90``` as the data source for pretraining, and then test the pretrained agent's LLDM performance on ```libero_10```.

## Pretrain on LIBERO-90
Replace ```GPU_ID``` with the cuda device ID. The trained agent will be saved into the ```./experiments_pretrained/``` folder.
```
export CUDA_VISIBLE_DEVICES=GPU_ID && export MUJOCO_EGL_DEVICE_ID=GPU_ID && python lifelong/main.py seed=SEED benchmark_name=BENCHMARK policy=POLICY lifelong=multitask pretrain=true train.num_workers=8
```

## Finetune on LIBERO-10
Replace the ```CKPT_PATH``` with the path to the saved checkpoint of the pretrained agent.
```
export CUDA_VISIBLE_DEVICES=GPU_ID && export MUJOCO_EGL_DEVICE_ID=GPU_ID && python lifelong/main.py seed=SEED benchmark_name=BENCHMARK policy=POLICY lifelong=ALGO pretrain_model_path=CKPT_PATH 
```
