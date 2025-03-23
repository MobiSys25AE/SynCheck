# Microbenchmark
We analyze the impact of the backbone models, original
synthetic data volume and generation conditions on task performance with SynCheck. The comparison baselines includes original CsiGAN/RF-Diffusion, filtering synthetic data with SSIM and filtering with TRTS. 

## CsiGAN

For all microbenchmark experiments with CsiGAN, the trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/dx7h8f5pkrgnjugdc8m52/AJE7RTi_E4-QO8t79iWK3Og?rlkey=493tmzkp49q6or1lfdtpi0tal&st=jsgs91ez&dl=0).

### Backbone model
Execute the following commands to run the experiments with different backbone model architectures for CsiGAN:
```bash
cd 5_microbenchmark/csigan
python run_backbone_csigan.py
```

1. **Original CsiGAN with CNN of mid-channel 64**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.806 | 0.668 | 0.764 | 0.710 | 0.702 | 0.730   |

2. **Original CsiGAN with CNN of mid-channel 256**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.714 | 0.798 | 0.754 | 0.758 | 0.768 | 0.758   |

3. **Filter with SSIM with CNN of mid-channel 64**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.798 | 0.682 | 0.764 | 0.762 | 0.708 | 0.743   |

4. **Filter with SSIM with CNN of mid-channel 256**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.768 | 0.770 | 0.748 | 0.748 | 0.726 | 0.752   |

5. **Filter with TRTS with CNN of mid-channel 64**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.780 | 0.804 | 0.838 | 0.812 | 0.810 | 0.809   |

6. **Filter with TRTS with CNN of mid-channel 256**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.774 | 0.764 | 0.796 | 0.800 | 0.800 | 0.787   |

7. **SynCheck with CNN of mid-channel 64**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.814 | 0.794 | 0.834 | 0.800 | 0.868 | 0.822   |

8. **SynCheck with CNN of mid-channel 256**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.784 | 0.822 | 0.810 | 0.820 | 0.848 | 0.817   |



### Synthetic data volume
To conduct the experiments with varying initial volumes of synthetic data for CsiGAN, follow the outlined procedure: First, generate an increased volume of synthetic data, and subsequently, train the task models using this expanded dataset. We provide the expanded synthetic dataset [here](https://www.dropbox.com/scl/fo/3i1pa38i6qrrw05g2curk/AAqAmrNTVUEPoUcAJR2wWSM?rlkey=cqewhgt4nh5mxhmjg3pqz6w3t&st=azgigpff&dl=0). Run the following commands for synthetic data generation and training: 
```bash
cd 1_synthetic_data_generation/csigan
python synthesize_more_data_csigan.py
cd 5_microbenchmark/csigan
python run_amount_csigan.py
```

1. **Original CsiGAN**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.710 | 0.816 | 0.808 | 0.746 | 0.762 | 0.768   |
| 1.0               | 0.700 | 0.804 | 0.780 | 0.774 | 0.752 | 0.762   |
| 1.5               | 0.774 | 0.800 | 0.772 | 0.760 | 0.792 | 0.780   |
| 2.0               | 0.710 | 0.774 | 0.806 | 0.770 | 0.792 | 0.770   |
| 2.5               | 0.742 | 0.792 | 0.782 | 0.794 | 0.760 | 0.774   |
| 3.0               | 0.758 | 0.834 | 0.770 | 0.774 | 0.780 | 0.783   |
| 3.5               | 0.780 | 0.810 | 0.798 | 0.776 | 0.780 | 0.789   |
| 4.0               | 0.718 | 0.828 | 0.816 | 0.790 | 0.812 | 0.793   |
| 4.5               | 0.730 | 0.818 | 0.824 | 0.756 | 0.802 | 0.786   |
| 5.0               | 0.748 | 0.824 | 0.770 | 0.790 | 0.784 | 0.783   |

2. **Filter with SSIM**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.778 | 0.800 | 0.802 | 0.778 | 0.812 | 0.794   |
| 1.0               | 0.682 | 0.782 | 0.802 | 0.726 | 0.732 | 0.745   |
| 1.5               | 0.744 | 0.784 | 0.788 | 0.780 | 0.756 | 0.770   |
| 2.0               | 0.742 | 0.808 | 0.794 | 0.754 | 0.762 | 0.772   |
| 2.5               | 0.776 | 0.812 | 0.760 | 0.788 | 0.772 | 0.782   |
| 3.0               | 0.720 | 0.798 | 0.810 | 0.756 | 0.800 | 0.777   |
| 3.5               | 0.812 | 0.822 | 0.778 | 0.750 | 0.760 | 0.784   |
| 4.0               | 0.754 | 0.752 | 0.764 | 0.788 | 0.812 | 0.774   |
| 4.5               | 0.762 | 0.836 | 0.814 | 0.768 | 0.776 | 0.791   |
| 5.0               | 0.774 | 0.810 | 0.762 | 0.782 | 0.738 | 0.773   |

3. **Filter with TRTS**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.782 | 0.798 | 0.794 | 0.774 | 0.772 | 0.784   |
| 1.0               | 0.848 | 0.818 | 0.820 | 0.780 | 0.784 | 0.810   |
| 1.5               | 0.804 | 0.748 | 0.842 | 0.780 | 0.790 | 0.793   |
| 2.0               | 0.764 | 0.776 | 0.798 | 0.808 | 0.800 | 0.789   |
| 2.5               | 0.816 | 0.796 | 0.838 | 0.780 | 0.734 | 0.793   |
| 3.0               | 0.820 | 0.784 | 0.832 | 0.816 | 0.796 | 0.810   |
| 3.5               | 0.802 | 0.754 | 0.818 | 0.788 | 0.780 | 0.788   |
| 4.0               | 0.818 | 0.770 | 0.840 | 0.814 | 0.762 | 0.801   |
| 4.5               | 0.812 | 0.836 | 0.830 | 0.774 | 0.822 | 0.815   |
| 5.0               | 0.804 | 0.772 | 0.828 | 0.778 | 0.752 | 0.787   |


4. **SynCheck**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.774 | 0.828 | 0.826 | 0.826 | 0.836 | 0.818   |
| 1.0               | 0.806 | 0.814 | 0.822 | 0.848 | 0.846 | 0.827   |
| 1.5               | 0.808 | 0.816 | 0.818 | 0.796 | 0.804 | 0.808   |
| 2.0               | 0.766 | 0.820 | 0.786 | 0.828 | 0.828 | 0.806   |
| 2.5               | 0.818 | 0.780 | 0.816 | 0.816 | 0.814 | 0.809   |
| 3.0               | 0.764 | 0.824 | 0.842 | 0.842 | 0.836 | 0.822   |
| 3.5               | 0.766 | 0.800 | 0.832 | 0.824 | 0.832 | 0.811   |
| 4.0               | 0.796 | 0.784 | 0.814 | 0.804 | 0.848 | 0.809   |
| 4.5               | 0.822 | 0.822 | 0.828 | 0.828 | 0.822 | 0.824   |
| 5.0               | 0.808 | 0.786 | 0.830 | 0.840 | 0.812 | 0.815   |


### Generation conditions
To examine the influence of generation conditions on both data synthesis and data utilization processes, we designed two distinct experiments. The first experiment employs unconditional generation: initially, an unconditional generative model is trained, and synthetic data is generated accordingly. Subsequently, the task model is trained using the original SynCheck framework on the unconditionally generated synthetic data. The second experiment involves filtering synthetic data while leveraging generation conditions as labels: here, we retain the data filtering mechanism of SynCheck but replace the assigned pseudo-labels with the generation conditions as labels during the training of the task model. Execute the following commands to run these experiments:
```bash
cd 1_synthetic_data_generation/csigan
python train_unconditional_csigan.py
python synthesize_unconditional_data_csigan.py
cd 5_microbenchmark/csigan
python run_condition_csigan.py
```
We also provide the checkpoint of unconditional generative model [here](https://www.dropbox.com/scl/fi/pvcdixpb48n7fp2gnn1g4/epoch4_l1trained0.7537_l1all0.7887.pth?rlkey=uyn0wft444s4i5c0pdbck2owv&st=fi2em21g&dl=0) and the processed unconditional synthetic data [here](https://www.dropbox.com/scl/fo/snmuy371c0z5toerivxon/AJNEW6peQsmmG6yUj5WBTyQ?rlkey=rknjko1tjzfw11gj62q4ui240&st=me0uk3nu&dl=0).


| Setup            | 1223  | 213   | 420   | 219   | 307   | Average |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| Uncond SynCheck  | 0.668 | 0.736 | 0.836 | 0.792 | 0.772 | 0.761   |
| CondLabel+Filter | 0.736 | 0.758 | 0.814 | 0.800 | 0.768 | 0.775   |


## RF-Diffusion

For all microbenchmark experiments with RF-Diffusion, the trained checkpoints from our experimental runs have been made available and can be accessed [here](https://www.dropbox.com/scl/fo/mlsv2cgyvkm3o7gnyp0g6/ACkwsLSk8441kK9G1W2nNQQ?rlkey=c3q1igozh206plgmysjtgtxlk&st=hjecnfnm&dl=0).

### Backbone model
Execute the following commands to run the experiments with different backbone model architectures for RF-Diffusion:
```bash
cd 5_microbenchmark/rf-diffusion
python run_backbone_rfdiffusion.py
```

1. **Original RF-Diffusion with ResNet18**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.701 | 0.718 | 0.647 | 0.681 | 0.720 | 0.693   |

2. **Original RF-Diffusion with ResNet50**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.567 | 0.566 | 0.569 | 0.567 | 0.587 | 0.571   |

3. **Filter with SSIM with ResNet18**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.721 | 0.716 | 0.721 | 0.706 | 0.719 | 0.717   |

4. **Filter with SSIM with ResNet50**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.584 | 0.606 | 0.595 | 0.616 | 0.575 | 0.595   |

5. **Filter with TRTS with ResNet18**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.711 | 0.725 | 0.722 | 0.723 | 0.736 | 0.723   |

6. **Filter with TRTS with ResNet50**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.622 | 0.598 | 0.615 | 0.612 | 0.637 | 0.617   |

7. **SynCheck with ResNet18**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.751 | 0.744 | 0.747 | 0.735 | 0.744 | 0.744   |

8. **SynCheck with ResNet50**

| random seeds | 1223  | 213   | 420   | 219   | 307   | Average |
| ------------ | ----- | ----- | ----- | ----- | ----- | ------- |
| Accuracy     | 0.686 | 0.678 | 0.701 | 0.662 | 0.657 | 0.677   |


### Synthetic data volume
To conduct the experiments with varying initial volumes of synthetic data for RF-Diffusion, follow the same procedure of synthetic data generation and task model training as in the CSIGAN experiment. We provide the expanded synthetic data [here](https://www.dropbox.com/scl/fi/vo4a7qkkff9nhu2e9ylcl/syn_more_data_native.zip?rlkey=v6d0t6isdy3kcxsidpofv86rl&st=mn8kxgwl&dl=0). Run the following commands for the experiments: 
```bash
cd 1_synthetic_data_generation/rf-diffusion
python synthesize_more_data_rfdiffusion.py
cd 5_microbenchmark/rf-diffusion
python run_amount_rfdiffusion.py
```

1. **Original RF-Diffusion**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.700 | 0.703 | 0.717 | 0.709 | 0.698 | 0.705   |
| 1.5               | 0.667 | 0.697 | 0.693 | 0.704 | 0.665 | 0.685   |
| 2.0               | 0.658 | 0.674 | 0.683 | 0.680 | 0.680 | 0.675   |
| 2.5               | 0.648 | 0.666 | 0.652 | 0.666 | 0.652 | 0.657   |
| 3.0               | 0.631 | 0.657 | 0.662 | 0.665 | 0.637 | 0.650   |
| 3.5               | 0.615 | 0.672 | 0.658 | 0.666 | 0.639 | 0.650   |
| 4.0               | 0.588 | 0.631 | 0.650 | 0.625 | 0.643 | 0.627   |
| 4.5               | 0.575 | 0.624 | 0.642 | 0.628 | 0.635 | 0.621   |
| 5.0               | 0.608 | 0.609 | 0.653 | 0.639 | 0.629 | 0.628   |

2. **Filter with SSIM**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.725 | 0.718 | 0.713 | 0.720 | 0.709 | 0.717   |
| 1.5               | 0.730 | 0.722 | 0.722 | 0.729 | 0.727 | 0.726   |
| 2.0               | 0.720 | 0.709 | 0.721 | 0.708 | 0.721 | 0.716   |
| 2.5               | 0.716 | 0.719 | 0.705 | 0.720 | 0.737 | 0.719   |
| 3.0               | 0.704 | 0.716 | 0.721 | 0.721 | 0.724 | 0.717   |
| 3.5               | 0.722 | 0.710 | 0.714 | 0.720 | 0.711 | 0.716   |
| 4.0               | 0.686 | 0.714 | 0.725 | 0.725 | 0.715 | 0.713   |
| 4.5               | 0.711 | 0.693 | 0.707 | 0.724 | 0.715 | 0.710   |
| 5.0               | 0.722 | 0.722 | 0.713 | 0.722 | 0.724 | 0.721   |

3. **Filter with TRTS**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.703 | 0.739 | 0.707 | 0.721 | 0.717 | 0.717   |
| 1.5               | 0.702 | 0.691 | 0.720 | 0.724 | 0.732 | 0.714   |
| 2.0               | 0.727 | 0.721 | 0.730 | 0.717 | 0.699 | 0.719   |
| 2.5               | 0.721 | 0.735 | 0.716 | 0.721 | 0.723 | 0.723   |
| 3.0               | 0.705 | 0.700 | 0.723 | 0.727 | 0.685 | 0.708   |
| 3.5               | 0.717 | 0.718 | 0.730 | 0.714 | 0.716 | 0.719   |
| 4.0               | 0.729 | 0.724 | 0.730 | 0.717 | 0.719 | 0.724   |
| 4.5               | 0.712 | 0.722 | 0.720 | 0.730 | 0.712 | 0.719   |
| 5.0               | 0.731 | 0.707 | 0.720 | 0.726 | 0.719 | 0.720   |

4. **SynCheck**

| Syn-to-real Ratio | 1223  | 213   | 420   | 219   | 307   | Average |
| ----------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| 0.5               | 0.751 | 0.756 | 0.757 | 0.756 | 0.755 | 0.755   |
| 1.5               | 0.759 | 0.759 | 0.754 | 0.761 | 0.778 | 0.762   |
| 2.0               | 0.747 | 0.751 | 0.760 | 0.766 | 0.757 | 0.756   |
| 2.5               | 0.759 | 0.750 | 0.760 | 0.761 | 0.756 | 0.757   |
| 3.0               | 0.764 | 0.748 | 0.751 | 0.755 | 0.755 | 0.755   |
| 3.5               | 0.761 | 0.748 | 0.758 | 0.762 | 0.766 | 0.759   |
| 4.0               | 0.745 | 0.748 | 0.761 | 0.762 | 0.761 | 0.756   |
| 4.5               | 0.751 | 0.753 | 0.744 | 0.757 | 0.761 | 0.753   |
| 5.0               | 0.749 | 0.756 | 0.758 | 0.760 | 0.760 | 0.756   |

### Generation conditions
We follow the same procedure as CsiGAN to examine the impact of generation conditions for RF-Diffusion. 
Run the following commands for the experiments:
```bash
cd 1_synthetic_data_generation/rf-diffusion
python train_rf_diffusion_unconditional.py
python synthesize_data_rfdiffusion_unconditional.py
cd 5_microbenchmark/rf-diffusion
python run_condition_rfdiffusion.py
```
We also provide the checkpoint of unconditional generative model [here](https://www.dropbox.com/scl/fi/rtwf5x6m3j1ipzwafj48l/3-loss0.0103.pth?rlkey=o3jggmd6ccbsve1pag0wbagla&st=pi16njpc&dl=0) and the processed unconditional synthetic data [here](https://www.dropbox.com/scl/fi/0uxf9qwzllio52r6mx7tx/uncond_syn_data_native.zip?rlkey=8z48s5l9bgkodfqw7m5wewe82&st=m8drm9y4&dl=0).


| Setup            | 1223  | 213   | 420   | 219   | 307   | Average |
| ---------------- | ----- | ----- | ----- | ----- | ----- | ------- |
| Uncond SynCheck  | 0.668 | 0.736 | 0.836 | 0.792 | 0.772 | 0.761   |
| CondLabel+Filter | 0.736 | 0.758 | 0.814 | 0.800 | 0.768 | 0.775   |