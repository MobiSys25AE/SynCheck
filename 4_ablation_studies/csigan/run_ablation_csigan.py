import subprocess
import os

# without ova
for seed in [1223, 213, 420, 219, 307]:
    subprocess.run(["python", "ablation_csigan.py", \
        "--seed", str(seed), "--normalize_csi", 
        "--lambda_ova", str(0.0), 
        "--checkpoint_root_dir", "csigan_ablation_checkpoints_woova"])

ckpt_root_dir = "csigan_ablation_checkpoints_woova"
ckpt_dirs = os.listdir(ckpt_root_dir)
woova_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    woova_acc_list.append(best_acc)

print(woova_acc_list)
print(sum(woova_acc_list)/len(woova_acc_list))


# without cons and ent
for seed in [1223, 213, 420, 219, 307]:
    subprocess.run(["python", "ablation_csigan.py", \
        "--seed", str(seed), "--normalize_csi", \
        "--lambda_socr", str(0.0), "--lambda_oem", str(0.0), 
        "--checkpoint_root_dir", "csigan_ablation_checkpoints_woconsent"])

ckpt_root_dir = "csigan_ablation_checkpoints_woconsent"
ckpt_dirs = os.listdir(ckpt_root_dir)
woconsent_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[1][:-4]) for f in model_ckpts])
    woconsent_acc_list.append(best_acc)

print(woconsent_acc_list)
print(sum(woconsent_acc_list)/len(woconsent_acc_list))

