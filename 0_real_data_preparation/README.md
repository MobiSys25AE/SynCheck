# Real data preparation

## CsiGAN data preparation
We have translated the MATLAB data processing code from [CsiGAN](https://github.com/ChunjingXiao/CsiGAN/blob/master/DatasetCodeForSignFi/gan_2unlab4_Cycle1use2_3_4_Cvariable_D200_30_ok_vary.m) into Python while preserving the original data processing logic. The dataset is structured as follows:

1. **Task Training Data**: Known users with labels (Subject 2 and Subject 3).  
2. **Task Training Data**: Unknown user (Subject 1), including 5 labeled gestures and 10 unlabeled gestures.  
3. **Task Testing Data**: Unknown user, encompassing all gestures.  
4. **CycleGAN Training Data**: Subject 2 and Subject 1 (5 labeled gestures).  
5. **CycleGAN Inference Data**: Subject 3 (all gestures).  

You can download the original [SignFi dataset](https://wm1693.box.com/s/kidoq54rv93ysojgzv7xjqixyzwir7lq), put the `dataset_lab_150.mat` in the `0_real_data_preparation` directory, and process it by following these steps:
```bash
cd 0_real_data_preparation
python csigan_data.py
```
Alternatively, you can directly download our pre-processed data from [here](https://www.dropbox.com/scl/fo/e5z3fuv23ivmj3i58c9xt/AAIDQxtztZuQIXu-foQ47H0?rlkey=9772jrk0n9tbto0ds2nqmtcr4&st=uvz7tjly&dl=0). Put the processed data in the `0_real_data_preparation/csigan_data` directory. The data directory should have the following structure:
```
csigan_data/
├── train_csi
├── train_label
├── train_leaveout_unlabeled_csi
├── train_leaveout_unlabeled_label
├── leaveout_test_csi
├── leaveout_test_label
├── ...
```

## RF-Diffusion data preparation
### CSI raw data
Since the RF-Diffusion data partition is not publicly available, we adopt the data partition scheme from [RFBoost](https://github.com/aiot-lab/RFBoost). This scheme utilizes a common subset of the Widar dataset, which includes the 6 most frequently performed gestures by 15 users across all rooms, resulting in a total of 11,250 samples.

You can download the original [Widar dataset](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset) along with the [RFBoost dataset partition](https://github.com/aiot-lab/RFBoost/tree/main/source/widar3/all_5500_top6_1000_2560). Put the data partition files in the `0_real_data_preparation/real_fname_label` directory. We also provide the partition files [here](https://www.dropbox.com/scl/fo/0c6m2w0ng7c6cufq1t11t/AN5ngBdnHa9cHuFq0O6FsQ0?rlkey=dnlfmghrl1w4ordph30hcp8it&st=obvdzxi8&dl=0).

### CSI data with condition
RF-Diffusion is a conditional generative model that requires training data in the format of `(CSI, condition)`. To extract the condition and store it alongside CSI in `.mat` files, run the provided script. You can adjust the input CSI directory within the script. 
```bash
cd 0_real_data_preparation
python extract_cond2mat.py
```
For convenience, we also offer pre-processed data [here](https://www.dropbox.com/scl/fi/rwviesmhzxyxq7p4wjmn1/cond_mat_CSI.zip?rlkey=o5bzt3jrpn3tc08kno1tem5cd&st=9u5vfc78&dl=0). Place the pre-processed data in the `0_real_data_preparation/cond_mat_CSI` directory.

**Note**: If you prefer not to train the RF-Diffusion generative model yourself, you can skip downloading the CSI data with conditions. Ready-to-use real and synthetic DFS data for the task model will be provided in the following sections.

### DFS task data
We follow the practice of RFBoost to convert raw Channel State Information (CSI) into Doppler Frequency Shift (DFS) features. We adapt the MATLAB-based DFS transformation code of Widar to Python, which is available in the `CSI2DFS.py` script. You can modify the input CSI directory and output DFS directory within the script. 
```bash
cd 0_real_data_preparation
python CSI2DFS.py
```
Alternatively, we provide pre-processed DFS data [here](https://www.dropbox.com/scl/fi/gbvepspehkvw6l8is69gz/real_dfs_data.zip?rlkey=l8fo79obdeo25gz1f2pcrmjr2&st=4e87g1gb&dl=0) for convenience. Put the processed DFS data in the `0_real_data_preparation/real_dfs_data` directory. 
The data directory should have the following structure:
```
real_dfs_data/
├── 20181130
│   ├── user5
│   │   ├── user5-1-1-1-1-r1.npz
│   │   ├── user5-1-1-1-1-r2.npz
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