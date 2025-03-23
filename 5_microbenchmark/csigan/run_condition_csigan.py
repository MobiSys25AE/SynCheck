import subprocess
import os

seed_list = [1223, 213, 420, 219, 307]    


for seed in seed_list:
    subprocess.run(["python", "unconditional_main_csigan.py", \
        "--seed", str(seed), "--normalize_csi"])    

ckpt_root_dir = "unconditional_csigan_syncheck_checkpoints"
ckpt_dirs = os.listdir(ckpt_root_dir)
syncheck_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    syncheck_acc_list.append(best_acc)

print(syncheck_acc_list)
print(sum(syncheck_acc_list)/len(syncheck_acc_list))


for seed in seed_list:
    subprocess.run(["python", "condlabel_main_csigan.py", \
        "--seed", str(seed), "--normalize_csi"])    

ckpt_root_dir = "condlabel_csigan_syncheck_checkpoints"
ckpt_dirs = os.listdir(ckpt_root_dir)
syncheck_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    syncheck_acc_list.append(best_acc)

print(syncheck_acc_list)
print(sum(syncheck_acc_list)/len(syncheck_acc_list))

