def keypoint_similarity(gt_kpts, pred_kpts, sigmas, areas):
    """
    Calculate Object Keypoint Similarity (OKS) for evaluation.
    """
    EPSILON = torch.finfo(torch.float32).eps
    dist_sq = (gt_kpts[:,None,:,0] - pred_kpts[...,0])**2 + (gt_kpts[:,None,:,1] - pred_kpts[...,1])**2
    vis_mask = gt_kpts[..., 2].int() > 0
    k = 2 * sigmas
    denom = 2 * (k**2) * (areas[:,None, None] + EPSILON)
    exp_term = dist_sq / denom
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (vis_mask[:, None, :].sum(-1) + EPSILON)
    return oks

def calculate_mean_of_dict_values(input_dict):
    """Helper to average OKS values in a dictionary."""
    if not input_dict:
        raise ValueError("Input dictionary is empty.")
    return sum(input_dict.values()) / len(input_dict)

def evaluate_oks(data_loader, model, device):
    """Compute mean OKS over all samples in a dataloader."""
    oks_dic = {}
    model.eval()
    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(imgs)
            flat_tensor_gt = targets[0]["keypoints"].reshape(-1, 3).to(device)
            main_subarrays = outputs[0]["keypoints"][:len(targets[0]["area"]), ...]
            flat_tensor_pd = main_subarrays.reshape(-1, 3).to(device)
            NUM_KPTs = len(flat_tensor_gt)
            KPTS_OKS_SIGMAS_UNIF = torch.ones(NUM_KPTs)/NUM_KPTs
            oks_scores = keypoint_similarity(flat_tensor_gt.unsqueeze(0), flat_tensor_pd.unsqueeze(0),
                                             sigmas=KPTS_OKS_SIGMAS_UNIF.to(device), areas=targets[0]["area"].to(device))
            oks_mean = torch.mean(oks_scores)
            oks_dic[targets[0]['image_id'].item()] = oks_mean.item()
    return calculate_mean_of_dict_values(oks_dic)

print("TRAIN OKS:", evaluate_oks(data_loader_train, model, device))
print("VAL OKS:", evaluate_oks(data_loader_val, model, device))
print("TEST OKS:", evaluate_oks(data_loader_test, model, device))
