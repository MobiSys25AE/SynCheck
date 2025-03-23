# %%
from cycler import cycler
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm
font_path = 'Arial.ttf'
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()
fm.fontManager.addfont(font_path)

colors = ['#084E87', '#ef8a00', '#267226', '#BF3F3F', '#414141', "#282828"]
save_dir = 'images'
os.makedirs(save_dir, exist_ok=True)
sys_name = 'SynCheck'

# %%
# comparison baselines: mix real/syn, filterssim, filtertrts
method_dir = 'rf-diffusion'
method_name = 'rfdiffusion'
baseline_dir = '../2_synthetic_data_utilization_baseline'
syncheck_dir = '../3_synthetic_data_utilization_SynCheck'

mix_acc = []
filterssim_acc = []
filtertrts_acc = []
syncheck_acc = []

ckpt_root_dir = os.path.join(baseline_dir, method_dir, f'{method_name}_onlyreal_checkpoints')
ckpt_dirs = os.listdir(ckpt_root_dir)
onlyreal_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    if method_name == 'csigan':
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    else:
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
    onlyreal_acc_list.append(best_acc)
onlyreal_acc = np.mean(onlyreal_acc_list)

ckpt_root_dir = os.path.join(baseline_dir, method_dir, f'{method_name}_reproduce_checkpoints')
ckpt_dirs = os.listdir(ckpt_root_dir)
mix_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    if method_name == 'csigan':
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    else:
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
    mix_acc_list.append(best_acc)
mix_acc.append(np.mean(mix_acc_list))

ckpt_root_dir = os.path.join(baseline_dir, method_dir, f'{method_name}_filterssim_checkpoints')
ckpt_dirs = os.listdir(ckpt_root_dir)
filterssim_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    if method_name == 'csigan':
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    else:
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
    filterssim_acc_list.append(best_acc)
filterssim_acc.append(np.mean(filterssim_acc_list))

# filtertrts
ckpt_root_dir = os.path.join(baseline_dir, method_dir, f'{method_name}_filtertrts_checkpoints')
ckpt_dirs = os.listdir(ckpt_root_dir)
filtertrts_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    if method_name == 'csigan':
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])  
    else:
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
    filtertrts_acc_list.append(best_acc)
filtertrts_acc.append(np.mean(filtertrts_acc_list))

# SynCheck
ckpt_root_dir = os.path.join(syncheck_dir, method_dir, f'{method_name}_syncheck_checkpoints')
ckpt_dirs = os.listdir(ckpt_root_dir)
syncheck_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    if method_name == 'csigan':
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    else:
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
    syncheck_acc_list.append(best_acc)
syncheck_acc.append(np.mean(syncheck_acc_list))

# for other syn-to-real ratio
baseline_dir = os.path.join('..', '5_microbenchmark')
syncheck_dir = os.path.join('..', '5_microbenchmark')
syn_ratio_list = [0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
for syn_ratio in syn_ratio_list:
    ckpt_root_dir = os.path.join(baseline_dir, method_dir, f'{method_name}_reproduce_checkpoints_{syn_ratio}')
    ckpt_dirs = os.listdir(ckpt_root_dir)
    mix_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        if method_name == 'csigan':
            best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        else:
            best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        mix_acc_list.append(best_acc)
    mix_acc.append(np.mean(mix_acc_list))

    ckpt_root_dir = os.path.join(baseline_dir, method_dir, f'{method_name}_filterssim_checkpoints_{syn_ratio}')
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filterssim_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        if method_name == 'csigan':
            best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        else:
            best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        filterssim_acc_list.append(best_acc)
    filterssim_acc.append(np.mean(filterssim_acc_list))

    # filtertrts
    ckpt_root_dir = os.path.join(baseline_dir, method_dir, f'{method_name}_filtertrts_checkpoints_{syn_ratio}')
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filtertrts_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        if method_name == 'csigan':
            best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])  
        else:
            best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        filtertrts_acc_list.append(best_acc)
    filtertrts_acc.append(np.mean(filtertrts_acc_list))

    # SynCheck
    ckpt_root_dir = os.path.join(syncheck_dir, method_dir, f'{method_name}_syncheck_checkpoints_{syn_ratio}')
    ckpt_dirs = os.listdir(ckpt_root_dir)
    syncheck_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        if method_name == 'csigan':
            best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        else:
            best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        syncheck_acc_list.append(best_acc)
    syncheck_acc.append(np.mean(syncheck_acc_list))

# change the first two elements of the list
ele1 = mix_acc[0]
mix_acc[0] = mix_acc[1]
mix_acc[1] = ele1

ele1 = filterssim_acc[0]
filterssim_acc[0] = filterssim_acc[1]
filterssim_acc[1] = ele1

ele1 = filtertrts_acc[0]
filtertrts_acc[0] = filtertrts_acc[1]
filtertrts_acc[1] = ele1

ele1 = syncheck_acc[0]
syncheck_acc[0] = syncheck_acc[1]
syncheck_acc[1] = ele1

mix_acc = [onlyreal_acc, ] + mix_acc
filterssim_acc = [onlyreal_acc, ] + filterssim_acc
filtertrts_acc = [onlyreal_acc, ] + filtertrts_acc
syncheck_acc = [onlyreal_acc, ] + syncheck_acc

print(mix_acc)
print(filterssim_acc)
print(filtertrts_acc)
print(syncheck_acc)


# %%
fig, ax = plt.subplots(figsize=(9,3), constrained_layout=True)
x = np.arange(len(syn_ratio_list) + 2)

plt.rcParams["figure.dpi"] = 80
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["savefig.dpi"] = 300

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.markeredgewidth'] = 3

plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = font_name

plt.plot(x, mix_acc, '-', marker = '.', color=colors[0], 
    markersize=6, linewidth=2, label = 'MixRealSyn', alpha=0.6)
plt.plot(x, filterssim_acc, '-', marker = '.', color=colors[1], 
    markersize=6, linewidth=2, label = 'FilterSSIM', alpha=0.8)
plt.plot(x, filtertrts_acc, '-', marker = '.', color=colors[2], 
    markersize=6, linewidth=2, label = 'FilterTRTS', alpha=0.8)
plt.plot(x, syncheck_acc, '-', marker = 'x', color=colors[3], 
    markersize=6, linewidth=3, label = sys_name, alpha=1)


plt.grid(linestyle='--', linewidth=0.5, zorder=0)
all_syn_ratio_list = [0.0, 0.5, 1.0] + syn_ratio_list[1:]
ax.set_xticks(x, [str(syn_ratio) for syn_ratio in all_syn_ratio_list])
ax.set_xlabel('Synthetic w.r.t Real Data Ratio', verticalalignment='top', fontsize=18)
ax.set_xlim([x[0]-0.1, x[-1]+0.1])

ax.set_ylabel('Accuracy', verticalalignment='bottom', fontsize=18)
ax.set_ylim([0.6, 0.8])
ax.set_yticks([0.60, 0.65, 0.7, 0.75, 0.8])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
ax.tick_params(axis='y', labelsize=14)

ax.legend(
    loc="upper center",
    frameon=False,
    bbox_to_anchor=(0, 1.05, 1, 0.15),
    ncol=5,
    borderaxespad=0,
    handlelength=1.5,
    mode="expand",
    handletextpad=0.5,
    columnspacing=1,
    labelspacing=0.3, 
)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_dir, f'amount_{method_name}.pdf'), dpi=800, bbox_inches='tight')
 



