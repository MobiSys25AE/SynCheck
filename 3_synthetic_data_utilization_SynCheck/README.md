# Synthetic Data Utilization with SynCheck

## CsiGAN
Execute the following command to conduct the experiments with SynCheck for CsiGAN:
```bash
cd 3_synthetic_data_utilization_SynCheck/csigan
python run_syncheck_csigan.py
```

In our experiments of CsiGAN, the hyperparameters for both the baselines and SynCheck were identical: a learning rate (lr) of 2e-4 with a linear decay of 0.1 every 40 epochs, a batch size of 16, and a total of 100 epochs.

The trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/11g6b7rt3de2ovwxd6qam/AOaFTS0IHpZZjlQsVDKkSTI?rlkey=vkuwln94ap5s9r1qa115i9054&st=yw613504&dl=0). The results of the experiments are detailed below:

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.806 | 0.814 | 0.822 | 0.848 | 0.846 | 0.827   |


## RF-Diffusion
Execute the following command to conduct the experiments with SynCheck for RF-Diffusion:
```bash
cd 3_synthetic_data_utilization_SynCheck/rf-diffusion
python run_syncheck_rfdiffusion.py
```

In our experiments of RF-diffusion, the hyperparameters for both the baselines and SynCheck were identical: a learning rate (lr) of 1e-3 with a linear decay of 0.9 every 2 epochs, a batch size of 64, and a total of 10 epochs.

The trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/px9czxb7bwovkixv3ofr8/ALgTKLuWze5FG-Y09tUFT-M?rlkey=xyilp115ctakxo5d843nomonk&st=apm9au7x&dl=0). The results of the experiments are detailed below:

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.762 | 0.751 | 0.756 | 0.751 | 0.751 | 0.756   |

