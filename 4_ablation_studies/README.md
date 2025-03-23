# Ablation Studies
We investigate the impact of individual loss components on the performance of SynCheck, including the one-versus-all loss for inlier-outlier detection, and the consistency loss and entropy loss for unlabeled synthetic data. 

## CsiGAN
Execute the following commands to run the ablation studies for CsiGAN:
```bash
cd 4_ablation_studies/csigan
python run_ablation_csigan.py
```

The trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/2ccvsujhkxuwq4l566rk8/AMDuYrpE2Hd0N2qY6fPrkss?rlkey=8lax1dccqkhci9zt106rgj7vi&st=46py5k39&dl=0). The results of the experiments are detailed below:

1. **Without one-versus-all loss** 

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.678 | 0.766 | 0.828 | 0.754 | 0.772 | 0.760   |

2. **Without consistency and entropy loss**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.684 | 0.738 | 0.804 | 0.790 | 0.736 | 0.750   |


## RF-Diffusion
Execute the following commands to run the ablation studies for RF-Diffusion:
```bash
cd 4_ablation_studies/rf-diffusion
python run_ablation_rfdiffusion.py
```

The trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/nf2h2pes4fv1s3mc70o7f/AIBuH_E9mKwO6isZVxD_rUg?rlkey=ls3wg94ydqqn6s279umezq725&st=hxpj07ja&dl=0). The results of the experiments are detailed below:

1. **Without one-versus-all loss** 

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.752 | 0.744 | 0.747 | 0.739 | 0.748 | 0.746   |

2. **Without consistency and entropy loss** 

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.755 | 0.750 | 0.750 | 0.745 | 0.751 | 0.751   |
