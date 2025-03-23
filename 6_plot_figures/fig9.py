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
# comparison baselines: only real, mix real/syn, filterssim, filtertrts
baseline_dir = '../2_synthetic_data_utilization_baseline'
syncheck_dir = '../3_synthetic_data_utilization_SynCheck'
method_dirs = ['csigan', 'rf-diffusion']
onlyreal_acc = []
mix_acc = []
filterssim_acc = []
filtertrts_acc = []
syncheck_acc = []

for method_dir in method_dirs:
    method_name = 'rfdiffusion' if method_dir == 'rf-diffusion' else 'csigan'
    # onlyreal
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
    onlyreal_acc.append(np.mean(onlyreal_acc_list))
    
    # mix
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
   
    # filterssim
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
    
    
    
print(onlyreal_acc)
print(mix_acc)
print(filterssim_acc)
print(filtertrts_acc)
print(syncheck_acc)



# %%
n_groups = 2
fig, ax = plt.subplots(figsize=(9,3), constrained_layout=True)


index = np.arange(n_groups)
bar_width = 0.1
interval=0.2
left_to_right_interval = [-0.3, -0.15, 0, 0.15, 0.3]

plt.rcParams["figure.dpi"] = 80
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["savefig.dpi"] = 300

plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 8
plt.rcParams['lines.markeredgewidth'] = 3

plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = font_name




rects1 = ax.bar(index + interval + bar_width + left_to_right_interval[0], onlyreal_acc, bar_width,
                color="#FFFFFF", edgecolor = colors[0],
                hatch='/' * 4, lw=2,
                label='OnlyReal')          

rects2 = ax.bar(index + interval + bar_width + left_to_right_interval[1], mix_acc, bar_width,
                color="#FFFFFF", edgecolor = colors[1],
                hatch='x' * 4, lw=2,
                label='MixRealSyn')    

rects3 = ax.bar(index + interval + bar_width + left_to_right_interval[2], filterssim_acc, bar_width,
                color="#FFFFFF", edgecolor = colors[2],
                hatch='\\' * 4, lw=2,
                label='FilterSSIM')   

rects4 = ax.bar(index + interval + bar_width + left_to_right_interval[3], filtertrts_acc, bar_width,
                color="#FFFFFF", edgecolor = colors[3],
                hatch='|' * 4, lw=2,
                label='FilterTRTS')

rects5 = ax.bar(index + interval + bar_width + left_to_right_interval[4], syncheck_acc, bar_width,
            color="#FFFFFF", edgecolor = colors[4],
            hatch='|' * 4, lw=2,
            label=sys_name)


# plt.grid(linestyle='--', linewidth=0.5, zorder=0, axis='y')
# Baseline
x = np.linspace(-0.1, 0.7, 100)
y = onlyreal_acc[0]*np.ones(100)
ax.plot(x, y, '--', color='000000', marker='None', zorder=10, linewidth=1,alpha=0.8)

x = np.linspace(0.9, 1.7, 100)
y = onlyreal_acc[1]*np.ones(100)
ax.plot(x, y, '--', color='000000', marker='None', zorder=10, linewidth=1,alpha=0.8)

ax.set_ylabel('Accuracy', verticalalignment='bottom', fontsize=18)
ax.set_xticks(index + interval + bar_width)
ax.set_xticklabels(('Cross-domain', 'In-domain'), fontsize=18)
ax.set_ylim([0.65, 0.85])
ax.set_yticks([0.65, 0.7, 0.75, 0.8, 0.85])
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
plt.savefig(os.path.join(save_dir, 'overall_perf.pdf'), dpi=800, bbox_inches='tight')


