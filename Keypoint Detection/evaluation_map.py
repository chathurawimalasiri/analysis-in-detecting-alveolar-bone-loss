# ==== Evaluation: mAP and OKS ====
from torchmetrics.detection import MeanAveragePrecision

def evaluate_mAP(data_loader, model, device):
    """
    Evaluate mean Average Precision (mAP) on a dataset.
    """
    map_metric = MeanAveragePrecision()
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            imgs, targets = batch
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(imgs)
            outputs = [{k: v.to(device) for k, v in out.items()} for out in outputs]
            map_metric.update(outputs, targets)
    return map_metric.compute()

print("mAP TRAIN:", evaluate_mAP(data_loader_train, model, device))
print("mAP VAL:", evaluate_mAP(data_loader_val, model, device))
print("mAP TEST:", evaluate_mAP(data_loader_test, model, device))
