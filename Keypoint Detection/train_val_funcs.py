# ==== Training and Validation Functions ====

def train_batch(batch, model, optim):
    """
    Run one training step.
    """
    imgs, targets = batch
    imgs = list(img.to(device) for img in imgs)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    optim.zero_grad()
    losses = model(imgs, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optim.step()
    return loss, losses

def validate_batch(batch, model):
    """
    Run one validation step (no gradient).
    """
    with torch.no_grad():
        imgs, targets = batch
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(imgs, targets)
        loss = sum(loss for loss in losses.values())
        return loss, losses
