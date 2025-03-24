# Synthetic data generation

## CsiGAN data generation
CsiGAN leverages the 5 gestures of a known user to transfer knowledge to the same gestures of an unknown user during GAN training. Subsequently, it utilizes all gestures from the known user to synthesize new gestures for the unknown user.

### Steps to Reproduce:
1. **Train the Generative Model**  
   Run the following command to train the CycleGAN model. 
   ```bash
   cd 1_synthetic_data_generation/csigan
   python train_cyclegan.py --normalize_csi
   ```
   We also provide trained checkpoint [here](https://www.dropbox.com/scl/fi/hpsit2cb3fnqddl5zatzv/epoch52_l1trained0.6895_l1all0.7516.pth?rlkey=9hdgxtpfhld8e66inctmfrkmr&st=zfriakvj&dl=0) and please place it in the `1_synthetic_data_generation/csigan/cyclegan_checkpoints/trained_ckpt` directory. 

2. **Generate Synthetic Data**  
   To synthesize new data, utilize the provided trained checkpoint and execute the following command. 
   ```bash
   cd 1_synthetic_data_generation/csigan
   python synthesize_data_csigan.py --normalize_csi
   ```
   We also provide synthetic data [here](https://www.dropbox.com/scl/fo/e5z3fuv23ivmj3i58c9xt/AAIDQxtztZuQIXu-foQ47H0?rlkey=9772jrk0n9tbto0ds2nqmtcr4&st=jc9octu1&dl=0). This link is the same as the one for the processed real data. If you have already downloaded the processed real data, both the real and synthetic data should be stored together in the `0_real_data_preparation/csigan_data` directory.

## RF-Diffusion data generation
We leverage the open-source implementation of [RF-Diffusion](https://github.com/mobicom24/RF-Diffusion) to generate synthetic data. For convenience, we also provide the training and inference code for the generative model in this repository.

### Steps to Reproduce:   
   
1. **Train the Generative Model**  
   Run the following command to train the RF-Diffusion model. 
   ```bash
   python train_rf_diffusion.py
   ```
   We also provide trained checkpoint [here](https://www.dropbox.com/scl/fi/3w9degbrr0nab6plwgwiz/5-loss0.0099.pth?rlkey=q0xxelea6j10ko5kfvn047wl2&st=5s57x8wo&dl=0) and please place it in the `1_synthetic_data_generation/rf-diffusion/rfdiffusion_checkpoints/trained_ckpt` directory.

2. **Generate Synthetic Data**  
   Use the provided trained checkpoint to synthesize new data. 
   ```bash
   python synthesize_data_rfdiffusion.py --normalize_dfs
   ```
   We also provide the synthetic data [here](https://www.dropbox.com/scl/fi/33wkwvy6lk6ul3l97nf39/syn_data_native.zip?rlkey=i8vuo04difjj3rptftzrlizq3&st=t9hv1xlp&dl=0). Place the pre-processed data in the `1_synthetic_data_generation/rf-diffusion/syn_data_native` directory. The `syn_data_native` directory has similar structure to the `0_real_data_preparation/real_dfs_data` directory:
   ```
   syn_data_native/
   ├── 20181130
   │   ├── user5
   │   │   ├── user5-1-1-1-1-r1.npz
   │   │   ├── user5-1-1-1-1-r3.npz
   │   │   ├── ...
   │   ├── user10
   │   ├── ...
   ├── 20181204
   │   ├── user1
   │   │   ├── user1-1-1-1-1-r1.npz
   │   │   ├── ...
   ├── 20181209
   └── 20181211
   ```



