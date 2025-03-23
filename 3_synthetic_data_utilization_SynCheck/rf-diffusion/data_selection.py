import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def exclude_dataset(args, dataset, model, device, exclude_known=False, logger=None):
    dataset.init_index()
    test_loader = DataLoader(dataset, batch_size=args.batch_size)
    model.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.arange(0, out_open.size(0)).long().cuda()
            pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0, pred_close]
            # outlier score less than 0.5
            known_ind = unk_score < 0.5
            if batch_idx == 0:
                known_all = known_ind
            else:
                known_all = torch.cat([known_all, known_ind], 0)

    known_all = known_all.data.cpu().numpy()
    if exclude_known:
        selected_indices = np.where(known_all == 0)[0]
    else:
        selected_indices = np.where(known_all != 0)[0]
    if logger is not None:
        logger_message = "selected ratio %s"%( (len(selected_indices)/ len(known_all)))
        logger.info(logger_message)
    model.train()
    dataset.set_index(selected_indices)