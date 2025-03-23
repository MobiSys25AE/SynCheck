import subprocess
import os

seed_list = [1223, 213, 420, 219, 307]    
channel_cnt_list = [64, 256]

for channel_cnt in channel_cnt_list:
    for seed in seed_list:
        subprocess.run(["python", "reproduce_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--mid_channels", str(channel_cnt), \
            "--checkpoint_root_dir", f"csigan_reproduce_checkpoints_channel{channel_cnt}"])    
 
    ckpt_root_dir = f"csigan_reproduce_checkpoints_channel{channel_cnt}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    reproduce_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        reproduce_acc_list.append(best_acc)
    
    print(reproduce_acc_list)
    print(sum(reproduce_acc_list)/len(reproduce_acc_list))

for channel_cnt in channel_cnt_list:
    for seed in seed_list:
        subprocess.run(["python", "filterssim_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--mid_channels", str(channel_cnt), \
            "--checkpoint_root_dir", f"csigan_filterssim_checkpoints_channel{channel_cnt}"])

    ckpt_root_dir = f"csigan_filterssim_checkpoints_channel{channel_cnt}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filterssim_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        filterssim_acc_list.append(best_acc)
    
    print(filterssim_acc_list)
    print(sum(filterssim_acc_list)/len(filterssim_acc_list))


for channel_cnt in channel_cnt_list:
    for seed in seed_list:
        subprocess.run(["python", "filtertrts_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--mid_channels", str(channel_cnt), \
            "--checkpoint_root_dir", f"csigan_filtertrts_checkpoints_channel{channel_cnt}"])    

    ckpt_root_dir = f"csigan_filtertrts_checkpoints_channel{channel_cnt}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    filtertrts_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        filtertrts_acc_list.append(best_acc)

    print(filtertrts_acc_list)
    print(sum(filtertrts_acc_list)/len(filtertrts_acc_list))


for channel_cnt in channel_cnt_list:
    for seed in seed_list:
        subprocess.run(["python", "main_csigan.py", \
            "--seed", str(seed), "--normalize_csi", \
            "--mid_channels", str(channel_cnt), \
            "--checkpoint_root_dir", f"csigan_syncheck_checkpoints_channel{channel_cnt}"])    

    ckpt_root_dir = f"csigan_syncheck_checkpoints_channel{channel_cnt}"
    ckpt_dirs = os.listdir(ckpt_root_dir)
    syncheck_acc_list = []
    for ckpt_dir in ckpt_dirs:
        model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
        best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
        syncheck_acc_list.append(best_acc)

    print(syncheck_acc_list)
    print(sum(syncheck_acc_list)/len(syncheck_acc_list))