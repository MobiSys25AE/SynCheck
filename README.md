# Artifact Evaluation of SynCheck

## Overview
We present **SynCheck**, a quality-guided framework designed to optimize the use of synthetic data in wireless sensing tasks. While generative models have demonstrated potential in augmenting real-world datasets, the inconsistent quality of synthetic data often results in unreliable performance improvements. SynCheck addresses this challenge by refining synthetic data quality during task model training, consistently outperforming traditional quality-oblivious approaches.

This repository provides our implementation of SynCheck, including code for synthetic data generation, as well as training scripts for baseline methods and SynCheck. The artifact evaluation is divided into 7 parts, with detailed instructions for each part available in the README files of the corresponding subdirectories.

## Prerequisites
To set up the environment, install PyTorch and other required dependencies using the following command:

```bash
conda create -n syncheck python=3.8
conda activate syncheck
pip install -r requirements.txt
```
A minimum of 200 GB of disk space is required to store the dataset. Furthermore, GPU support is highly recommended to enhance computational efficiency and accelerate the training process.

## Part 0: Real Data Preparation  
Please refer to the `0_real_data_preparation` subdirectory. It details the procedures for preparing the real-world dataset of CsiGAN and RF-Diffusion.  

## Part 1: Synthetic Data Generation  
Please refer to the `1_synthetic_data_generation` subdirectory. It details the steps for generating synthetic data using the CsiGAN and RF-Diffusion models.  

## Part 2: Baseline Methods  
Please refer to the `2_synthetic_data_utilization_baseline` subdirectory. It implements the baseline methods, which are trained on both the real data prepared in Part 0 and the synthetic data generated in Part 1.  

## Part 3: SynCheck Method  
Please refer to the `3_synthetic_data_utilization_SynCheck` subdirectory. It implements the SynCheck method, which is trained on real data and a selected subset of synthetic data.

## Part 4: Ablation Studies  
Please refer to the `4_ablation_studies` subdirectory. It details the necessity of different loss components for the SynCheck method.

## Part 5: Microbenchmark
Please refer to the `5_microbenchmark` subdirectory. It explores the impact of different components (backbone model, synthetic data volume and generation conditions) on the SynCheck method.

## Part 6: Plot Figures
Please refer to the `6_plot_figures` subdirectory. It contains scripts to visualize the experimental results and generate the figures presented in the paper.