import numpy as np
import scipy.io
import os

# Load the dataset
data = scipy.io.loadmat('dataset_lab_150.mat')
label = data['label']
csi1 = data['csi1']
csi2 = data['csi2']
csi3 = data['csi3']
csi4 = data['csi4']
csi5 = data['csi5']
csi1 = np.abs(csi1)
csi2 = np.abs(csi2)
csi3 = np.abs(csi3)
csi4 = np.abs(csi4)
csi5 = np.abs(csi5)

# Process labels
label1 = np.concatenate((label[0:1250], label[1250:1500] - 128))
label2 = np.concatenate((label[1500:2750], label[2750:3000] - 128))
label3 = np.concatenate((label[3000:4250], label[4250:4500] - 128))
label4 = np.concatenate((label[4500:5750], label[5750:6000] - 128))
label5 = np.concatenate((label[6000:7250], label[7250:7500] - 128))

# Define known and unknown sets
trainCsi1 = csi2
trainlabel1 = label2
trainCsi2 = csi3
trainlabel2 = label3
trainCsi3 = csi4
trainlabel3 = label4
trainCsi4 = csi5
trainlabel4 = label5

leaveCsi = csi1
leaveLabel = label1

cycleCsi_user2 = trainCsi2
cycleLabelUser2 = trainlabel2

select_category = 50
unlabeled_category = 10
cycle_category = 5
train_cnt_per_category = 5

# training CSI data: 2 known users
train_csi_all = np.concatenate((trainCsi1, trainCsi2), axis=3)
train_label_all = np.concatenate((trainlabel1, trainlabel2))
leave_csi_all = leaveCsi
leave_label_all = leaveLabel

# select `select_category` gestures fortraining with 2 known users
train_csi = []
train_label = []
for i in range(1, select_category + 1):
    train_indices = np.where(train_label_all == i)[0]
    train_csi.append(train_csi_all[:, :, :, train_indices])
    train_label.extend(train_label_all[train_indices])
train_csi = np.concatenate(train_csi, axis=3)
train_label = np.array(train_label)

# `select_category` gestures of leaveout user for testing
leaveout_test_csi = []
leaveout_test_label = []
for i in range(1, select_category + 1):
    leaveout_indices = np.where(leave_label_all == i)[0]
    leaveout_test_csi.append(leave_csi_all[:, :, :, leaveout_indices])
    leaveout_test_label.extend(leave_label_all[leaveout_indices])
leaveout_test_csi = np.concatenate(leaveout_test_csi, axis=3)
leaveout_test_label = np.array(leaveout_test_label)

# for `unlabeled_category` gestures
# use [0:train_cnt_per_category] of leaveout user for training
train_leaveout_unlabeled_csi = []
train_leaveout_unlabeled_label = []
for i in range(1, unlabeled_category + 1):
    leaveout_indices = np.where(leave_label_all == i)[0]
    train_leaveout_unlabeled_csi.append(leave_csi_all[:, :, :, leaveout_indices[0:train_cnt_per_category]])
    train_leaveout_unlabeled_label.extend(leave_label_all[leaveout_indices[0:train_cnt_per_category]])
train_leaveout_unlabeled_csi = np.concatenate(train_leaveout_unlabeled_csi, axis=3)
train_leaveout_unlabeled_label = np.array(train_leaveout_unlabeled_label)

# for `cycle_category` gestures
# use [0:train_cnt_per_category] of user2 for cycleGAN training source domain
cycle_source_csi = []
cycle_source_label = []
for i in range(1, cycle_category + 1):
    user2_indices = np.where(cycleLabelUser2 == i)[0]
    cycle_source_csi.append(cycleCsi_user2[:, :, :, user2_indices[0:train_cnt_per_category]])
    cycle_source_label.extend(cycleLabelUser2[user2_indices[0:train_cnt_per_category]])
cycle_source_csi = np.concatenate(cycle_source_csi, axis=3)
cycle_source_label = np.array(cycle_source_label)

# for `select_category` gestures
# use [0:train_cnt_per_category] of user2 for cycleGAN all source data (inference to generate data)
cycle_source_all_csi = []
cycle_source_all_label = []
for i in range(1, select_category + 1):
    user2_indices = np.where(cycleLabelUser2 == i)[0]
    cycle_source_all_csi.append(cycleCsi_user2[:, :, :, user2_indices[0:train_cnt_per_category]])
    cycle_source_all_label.extend(cycleLabelUser2[user2_indices[0:train_cnt_per_category]])
cycle_source_all_csi = np.concatenate(cycle_source_all_csi, axis=3)
cycle_source_all_label = np.array(cycle_source_all_label)

# for `cycle_category` gestures
# use [0:train_cnt_per_category] of leaveout user for cycleGAN training target domain
cycle_target_csi = []
cycle_target_label = []
for i in range(1, cycle_category + 1):
    leaveout_indices = np.where(leave_label_all == i)[0]
    cycle_target_csi.append(leave_csi_all[:, :, :, leaveout_indices[0:train_cnt_per_category]])
    cycle_target_label.extend(leave_label_all[leaveout_indices[0:train_cnt_per_category]])
cycle_target_csi = np.concatenate(cycle_target_csi, axis=3)
cycle_target_label = np.array(cycle_target_label)

# for `select_category` gestures
# use [0:train_cnt_per_category] of leaveout user for cycleGAN target domain all
cycle_target_all_csi = []
cycle_target_all_label = []
for i in range(1, select_category + 1):
    leaveout_indices = np.where(leave_label_all == i)[0]
    cycle_target_all_csi.append(leave_csi_all[:, :, :, leaveout_indices[0:train_cnt_per_category]])
    cycle_target_all_label.extend(leave_label_all[leaveout_indices[0:train_cnt_per_category]])
cycle_target_all_csi = np.concatenate(cycle_target_all_csi, axis=3)
cycle_target_all_label = np.array(cycle_target_all_label)


# reshape arrays from (200, 30, 3, bs) to (bs, 200, 30, 3)
train_csi = np.transpose(train_csi, (3, 0, 1, 2))
train_label = np.array([i-1 for i in train_label])
train_label = train_label.squeeze()
leaveout_test_csi = np.transpose(leaveout_test_csi, (3, 0, 1, 2))
leaveout_test_label = np.array([i-1 for i in leaveout_test_label])
leaveout_test_label = leaveout_test_label.squeeze()
train_leaveout_unlabeled_csi = np.transpose(train_leaveout_unlabeled_csi, (3, 0, 1, 2))
train_leaveout_unlabeled_label = np.array([i-1 for i in train_leaveout_unlabeled_label])
train_leaveout_unlabeled_label = train_leaveout_unlabeled_label.squeeze()
cycle_source_csi = np.transpose(cycle_source_csi, (3, 0, 1, 2))
cycle_source_label = np.array([i-1 for i in cycle_source_label])
cycle_source_label = cycle_source_label.squeeze()
cycle_source_all_csi = np.transpose(cycle_source_all_csi, (3, 0, 1, 2))
cycle_source_all_label = np.array([i-1 for i in cycle_source_all_label])
cycle_source_all_label = cycle_source_all_label.squeeze()
cycle_target_csi = np.transpose(cycle_target_csi, (3, 0, 1, 2))
cycle_target_label = np.array([i-1 for i in cycle_target_label])
cycle_target_label = cycle_target_label.squeeze()
cycle_target_all_csi = np.transpose(cycle_target_all_csi, (3, 0, 1, 2))
cycle_target_all_label = np.array([i-1 for i in cycle_target_all_label])
cycle_target_all_label = cycle_target_all_label.squeeze()

# print sizes
print(f'size(train_csi): {train_csi.shape}')
print(f'size(train_label): {train_label.shape}')
print(f'size(leaveout_test_csi): {leaveout_test_csi.shape}')
print(f'size(leaveout_test_label): {leaveout_test_label.shape}')
print(f'size(train_leaveout_unlabeled_csi): {train_leaveout_unlabeled_csi.shape}')
print(f'size(train_leaveout_unlabeled_label): {train_leaveout_unlabeled_label.shape}')
print(f'size(cycle_source_csi): {cycle_source_csi.shape}')
print(f'size(cycle_source_label): {cycle_source_label.shape}')
print(f'size(cycle_source_all_csi): {cycle_source_all_csi.shape}')
print(f'size(cycle_source_all_label): {cycle_source_all_label.shape}')
print(f'size(cycle_target_csi): {cycle_target_csi.shape}')
print(f'size(cycle_target_label): {cycle_target_label.shape}')
print(f'size(cycle_target_all_csi): {cycle_target_all_csi.shape}')
print(f'size(cycle_target_all_label): {cycle_target_all_label.shape}')

# save data
dst_dir = 'csigan_data'
os.makedirs(dst_dir, exist_ok=True)
np.save(os.path.join(dst_dir, 'train_csi.npy'), train_csi)
np.save(os.path.join(dst_dir, 'train_label.npy'), train_label)
np.save(os.path.join(dst_dir, 'leaveout_test_csi.npy'), leaveout_test_csi)
np.save(os.path.join(dst_dir, 'leaveout_test_label.npy'), leaveout_test_label)
np.save(os.path.join(dst_dir, 'train_leaveout_unlabeled_csi.npy'), train_leaveout_unlabeled_csi)
np.save(os.path.join(dst_dir, 'train_leaveout_unlabeled_label.npy'), train_leaveout_unlabeled_label)
np.save(os.path.join(dst_dir, 'cycle_source_csi.npy'), cycle_source_csi)
np.save(os.path.join(dst_dir, 'cycle_source_label.npy'), cycle_source_label)
np.save(os.path.join(dst_dir, 'cycle_source_all_csi.npy'), cycle_source_all_csi)
np.save(os.path.join(dst_dir, 'cycle_source_all_label.npy'), cycle_source_all_label)
np.save(os.path.join(dst_dir, 'cycle_target_csi.npy'), cycle_target_csi)
np.save(os.path.join(dst_dir, 'cycle_target_label.npy'), cycle_target_label)
np.save(os.path.join(dst_dir, 'cycle_target_all_csi.npy'), cycle_target_all_csi)
np.save(os.path.join(dst_dir, 'cycle_target_all_label.npy'), cycle_target_all_label)

