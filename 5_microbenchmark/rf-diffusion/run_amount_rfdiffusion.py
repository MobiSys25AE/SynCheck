import subprocess
import os

seed_list = [1223, 213, 420, 219, 307]   
syn_ratio_list = [0.5, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

for syn_ratio in syn_ratio_list:
    for seed in seed_list:
        subprocess.run(["python", "filterssim_rfdiffusion.py", \
            "--seed", str(seed), "--normalize_dfs", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"rfdiffusion_filterssim_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"rfdiffusion_filterssim_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filterssim_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        filterssim_acc_list.append(best_acc)

    print(filterssim_acc_list)
    print(sum(filterssim_acc_list)/len(filterssim_acc_list))


for syn_ratio in syn_ratio_list:
    for seed in seed_list:
        subprocess.run(["python", "filtertrts_rfdiffusion.py", \
            "--seed", str(seed), "--normalize_dfs", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"rfdiffusion_filtertrts_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"rfdiffusion_filtertrts_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filtertrts_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        filtertrts_acc_list.append(best_acc)

    print(filtertrts_acc_list)
    print(sum(filtertrts_acc_list)/len(filtertrts_acc_list))


for syn_ratio in syn_ratio_list:
    for seed in seed_list:
        subprocess.run(["python", "reproduce_rfdiffusion.py", \
            "--seed", str(seed), "--normalize_dfs", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"rfdiffusion_reproduce_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"rfdiffusion_reproduce_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    reproduce_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        reproduce_acc_list.append(best_acc)

    print(reproduce_acc_list)
    print(sum(reproduce_acc_list)/len(reproduce_acc_list))


for syn_ratio in syn_ratio_list:
    for seed in seed_list:
        subprocess.run(["python", "main_rfdiffusion.py", \
            "--seed", str(seed), "--normalize_dfs", \
            "--syn_ratio", str(syn_ratio), \
            "--checkpoint_root_dir", f"rfdiffusion_syncheck_checkpoints_{syn_ratio}"])

    ckpt_root_dir = f"rfdiffusion_syncheck_checkpoints_{syn_ratio}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    syncheck_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
        syncheck_acc_list.append(best_acc)

    print(syncheck_acc_list)
    print(sum(syncheck_acc_list)/len(syncheck_acc_list))

