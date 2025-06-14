# ==== Model Saving ====
save_path = 'keypointRCNN_script_report_04_09__01.pth'
torch.save(model, save_path)
print(f"Model saved to {save_path}")
