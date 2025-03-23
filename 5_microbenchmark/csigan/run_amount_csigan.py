import subprocess
import os

seed_list = [1223, 213, 420, 219, 307]    
syn_ratio_list = [0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

mean_reproduce_list = []
for syn_ratio in syn_ratio_list:
    for seed in seed_list:
        subprocess.run(["python", "reproduce_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"csigan_reproduce_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"csigan_reproduce_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    reproduce_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        reproduce_acc_list.append(best_acc)

    print(reproduce_acc_list)
    print(syn_ratio, 'reproduce', sum(reproduce_acc_list)/len(reproduce_acc_list))
    mean_reproduce_list.append(sum(reproduce_acc_list)/len(reproduce_acc_list))

print(mean_reproduce_list)

mean_ssim_list = []
for syn_ratio in syn_ratio_list:
    for seed in seed_list:
        subprocess.run(["python", "filterssim_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"csigan_filterssim_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"csigan_filterssim_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filterssim_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        filterssim_acc_list.append(best_acc)
    
    print(filterssim_acc_list)
    print(syn_ratio, 'filterssim', sum(filterssim_acc_list)/len(filterssim_acc_list))
    mean_ssim_list.append(sum(filterssim_acc_list)/len(filterssim_acc_list))

print(mean_ssim_list)

mean_trts_list = []
for syn_ratio in syn_ratio_list:
    for seed in seed_list:
        subprocess.run(["python", "filtertrts_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"csigan_filtertrts_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"csigan_filtertrts_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filtertrts_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        filtertrts_acc_list.append(best_acc)

    print(filtertrts_acc_list)
    print(syn_ratio, 'filtertrts', sum(filtertrts_acc_list)/len(filtertrts_acc_list))
    mean_trts_list.append(sum(filtertrts_acc_list)/len(filtertrts_acc_list))

print(mean_trts_list)

mean_syncheck_list = []
for syn_ratio in syn_ratio_list:
    for seed in [1223, 213, 420, 219, 307]:
        subprocess.run(["python", "main_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"csigan_syncheck_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"csigan_syncheck_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    syncheck_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        syncheck_acc_list.append(best_acc)

    print(syncheck_acc_list)
    print(syn_ratio, 'syncheck', sum(syncheck_acc_list)/len(syncheck_acc_list))
    mean_syncheck_list.append(sum(syncheck_acc_list)/len(syncheck_acc_list))

print(mean_syncheck_list)