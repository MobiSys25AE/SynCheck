import subprocess
import os

seed_list = [1223, 213, 420, 219, 307]

for seed in seed_list:
    subprocess.run(["python", "onlyreal_csigan.py", \
        "--seed", str(seed), "--normalize_csi"])

ckpt_root_dir = 'csigan_onlyreal_checkpoints'
ckpt_dirs = os.listdir(ckpt_root_dir)
onlyreal_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    onlyreal_acc_list.append(best_acc)


for seed in seed_list:
    subprocess.run(["python", "reproduce_csigan.py", \
        "--seed", str(seed), "--normalize_csi"])

ckpt_root_dir = 'csigan_reproduce_checkpoints'
ckpt_dirs = os.listdir(ckpt_root_dir)
reproduce_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    reproduce_acc_list.append(best_acc)
    

for seed in seed_list:
    subprocess.run(["python", "filterssim_csigan.py", \
        "--seed", str(seed), "--normalize_csi"])

ckpt_root_dir = 'csigan_filterssim_checkpoints'
ckpt_dirs = os.listdir(ckpt_root_dir)
filterssim_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    filterssim_acc_list.append(best_acc)


for seed in seed_list:
    subprocess.run(["python", "filtertrts_csigan.py", \
        "--seed", str(seed), "--normalize_csi"])

ckpt_root_dir = 'csigan_filtertrts_checkpoints'
ckpt_dirs = os.listdir(ckpt_root_dir)
filtertrts_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    filtertrts_acc_list.append(best_acc)


print(onlyreal_acc_list)
print(sum(onlyreal_acc_list)/len(onlyreal_acc_list))

print(reproduce_acc_list)
print(sum(reproduce_acc_list)/len(reproduce_acc_list))

print(filterssim_acc_list)
print(sum(filterssim_acc_list)/len(filterssim_acc_list))

print(filtertrts_acc_list)
print(sum(filtertrts_acc_list)/len(filtertrts_acc_list))
