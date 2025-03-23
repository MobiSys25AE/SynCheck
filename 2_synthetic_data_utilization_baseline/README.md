# Synthetic data utilization baseline

## CsiGAN
The original CsiGAN implementation combines both CycleGAN and a vanilla GAN, with the training of these two GAN models conducted concurrently during task model training. In our implementation, we decouple this process by extracting the CycleGAN training to generate synthetic data in advance (Stage 1: Synthetic Data Generation). 

Execute the following command to conduct the baseline experiments, which include the original CsiGAN, only real data, filtering synthetic datawith SSIM, and filtering with TRTS:
```bash
cd 2_synthetic_data_utilization_baseline/csigan
python run_baseline_csigan.py
```
**Note**: There are two primary reasons for the differences in results:

1. **Vanilla GAN in CsiGAN**: CsiGAN utilizes a vanilla GAN to generate low-quality synthetic data during task model training. The random seed for this process is intentionally not fixed, ensuring that different synthetic data is generated in each epoch.  
2. **Random Seed Averaging**: In our implementation, we use 5 random seeds and average the results, while the original implementation uses a single random seed. 

In our experiments of CsiGAN, the hyperparameters for both the baselines and SynCheck were identical: a learning rate (lr) of 2e-4 with a linear decay of 0.1 every 40 epochs, a batch size of 16, and a total of 100 epochs.

The trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/pncxhuwzi80u1zmf2ujck/AJUXWac7cAKEAtuZU3zwIns?rlkey=d2q3nw89gcwbq5s6i1isjzdr6&st=anfjoo6q&dl=0). The results of the experiments are detailed below:

1. **Only real method**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.680 | 0.798 | 0.696 | 0.724 | 0.760 | 0.732   |

2. **CsiGAN method**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.700 | 0.804 | 0.780 | 0.774 | 0.752 | 0.762   |

3. **Filter with SSIM**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.682 | 0.782 | 0.802 | 0.726 | 0.732 | 0.745   |

4. **Filter with TRTS**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.848 | 0.818 | 0.820 | 0.780 | 0.784 | 0.810   |


## RF-Diffusion
Execute the following command to conduct the baseline experiments, which include the original CsiGAN, only real data, filtering synthetic datawith SSIM, and filtering with TRTS:
```bash
cd 2_synthetic_data_utilization_baseline/rf-diffusion
python run_baseline_rfdiffusion.py
```

In our experiments of RF-diffusion, the hyperparameters for both the baselines and SynCheck were identical: a learning rate (lr) of 1e-3 with a linear decay of 0.9 every 2 epochs, a batch size of 64, and a total of 10 epochs.

The trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/2xdnla00co0fvlig2npfb/AInI9ByGyJyIRS6lvf9fbNk?rlkey=f5npbfjejvpzks4nvp9yw5xgj&st=lce14czk&dl=0). The results of the experiments are detailed below:

1. **Only real method**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.733 | 0.713 | 0.716 | 0.744 | 0.719 | 0.725   |

2. **RF-Diffusion method**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.720 | 0.699 | 0.704 | 0.718 | 0.686 | 0.705   |

3. **Filter with SSIM**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.716 | 0.725 | 0.727 | 0.738 | 0.726 | 0.726   |

4. **Filter with TRTS**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.717 | 0.719 | 0.740 | 0.734 | 0.739 | 0.730   |

