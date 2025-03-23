import subprocess
import os

for seed in [1223, 213, 420, 219, 307]:
    subprocess.run(["python", "main_rfdiffusion.py", \
        "--seed", str(seed), "--normalize_dfs"])

ckpt_root_dir = 'rfdiffusion_syncheck_checkpoints'
ckpt_dirs = os.listdir(ckpt_root_dir)
syncheck_acc_list = []
for ckpt_dir in ckpt_dirs:
    model_ckpts = [f for f in os.listdir(os.path.join(ckpt_root_dir, ckpt_dir)) if f.endswith('.pth')]
    best_acc = max([float(f.split('_')[0][4:]) for f in model_ckpts])
    syncheck_acc_list.append(best_acc)

print(syncheck_acc_list)
print(sum(syncheck_acc_list)/len(syncheck_acc_list))