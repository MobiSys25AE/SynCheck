import torch
import torch.nn.functional as F

def ova_loss(logits_open, label):
    # (bs, 2, num_classes)
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_inlier = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.arange(0, logits_open.size(0)).long()
    label_inlier[label_range, label] = 1
    label_outlier = 1 - label_inlier
    # inlier
    loss_inlier = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_inlier, 1))
    # outlier
    loss_outlier = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_outlier, 1)[0])
    Lo = loss_outlier + loss_inlier
    return Lo 


def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(
        -logits_open * torch.log(logits_open + 1e-8), 1), 1))
    return Le