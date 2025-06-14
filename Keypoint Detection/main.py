from data_augmentation import train_transform, val_transform
from model import get_model
from data_prep import get_data_loaders  
from train_loop import train_model
from save_model import save_model
from evaluation_map import evaluate_mAP
from evaluation_oks import evaluate_oks
from logging_utils import log_message

import torch

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = get_data_loaders(train_transform, val_transform)  # <-- Pass transforms here!
    model = get_model(num_keypoints=)
    model.to(device)

    # Train model
    train_model(model, data_loaders, device)
    # Save model
    save_model(model, "model.pth")
    print("Model saved as model.pth.")
    # Evaluate
    print("mAP:", evaluate_mAP(data_loaders['val'], model, device))
    print("OKS:", evaluate_oks(data_loaders['val'], model, device))
    # Log result
    log_message("Training complete.")

if __name__ == "__main__":
    main()
