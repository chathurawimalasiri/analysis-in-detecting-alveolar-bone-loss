#!/usr/bin/env python
# coding: utf-8


# # 1. Imports

# In[1]:


import os, json, numpy as np, matplotlib.pyplot as plt
import cv2
from PIL import Image
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F

import albumentations as A # Library for augmentation
import transforms


# Clear GPU cache
torch.cuda.empty_cache()


# # 2. Augmentations

# # Get Mean and Std

# In[2]:


# from PIL import Image
# import os
# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader

# class CustomDataset(Dataset):
#     def __init__(self, file_paths, transform=None):
#         self.file_paths = file_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.file_paths[idx])
#         if self.transform:
#             image = self.transform(image)
#         return image


# # Define the transformation without normalization
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # Replace 'x_val.values' with the path to your image files in the 'images' folder
# KEYPOINTS_FOLDER_TRAIN = 'Data/test/images'
# file_paths = [os.path.join(KEYPOINTS_FOLDER_TRAIN, filename) for filename in os.listdir(KEYPOINTS_FOLDER_TRAIN)]

# # Create a custom dataset
# custom_dataset = CustomDataset(file_paths, transform=transform)

# # Create a DataLoader with the custom collate function
# batch_size = 1
# loader = DataLoader(custom_dataset, batch_size=batch_size)

# # Calculate mean and std
# data = next(iter(loader))
# mean = data.mean(dim=(0, 2, 3))
# std = data.std(dim=(0, 2, 3))

# print("Mean:", mean)
# print("Std:", std)


# In[7]:


def train_transform():
    return A.Compose([
        A.Sequential([
           # A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),  # Random change of brightness & contrast
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.1),
            A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
            #A.Rotate(limit=30, p=0.25),
            #A.Normalize(mean=(0.5532, 0.5532, 0.5532), std=(0.2804, 0.2805, 0.2804)),
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
    )


# In[8]:


def val_transform():
    return A.Compose([
        A.Sequential([
            #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),  # Random change of brightness & contrast
            #A.HorizontalFlip(p=0.25),
            #A.VerticalFlip(p=0.25),
            A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
            #A.Rotate(limit=30, p=0.5),
            #A.Normalize(mean=(0.5828, 0.5829, 0.5828), std=(0.2437, 0.2438, 0.2437)),
        ], p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
    )


# In[ ]:





# # 3. Dataset class

# In[5]:


class ClassDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo # Use demo=True if you need transformed and original images (for example, for visualization purposes)
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        annotations_path = os.path.join(self.root, "annotations", self.annotations_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)


        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = data['bboxes']
            keypoints_original = data['keypoints']

            #keypoints_original = [[subsub for subsub in sublist[:2]] for sublist in keypoints_original]  # ONLY CEJ
            #keypoints_original = [[subsub for subsub in sublist[2:4]] for sublist in keypoints_original]  # ONLY AEAC
            keypoints_original = [[subsub for subsub in sublist[4:]] for sublist in keypoints_original]  # ONLY APEX

            keypoints_original = [sorted(sublist, key=lambda x: x[0]) for sublist in keypoints_original]

            # # All objects are teeth
            # bboxes_labels_original = ['teeth' for _ in bboxes_original]
            bboxes_labels_original = data['labels']
        if self.transform:
            # Converting keypoints from [x,y,visibility]-fiormat to [x, y]-format + Flattening nested list of keypoints
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2]
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]


            # Apply augmentations
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original, keypoints=keypoints_original_flattened)
            img = transformed['image']
            bboxes = transformed['bboxes']


            # Unflattening list transformed['keypoints']
            # For example, if we have the following list of keypoints for three objects (each object has two keypoints):
            # [obj1_kp1, obj1_kp2, obj2_kp1, obj2_kp2, obj3_kp1, obj3_kp2], where each keypoint is in [x, y]-format
            # Then we need to convert it to the following list:
            # [[obj1_kp1, obj1_kp2], [obj2_kp1, obj2_kp2], [obj3_kp1, obj3_kp2]]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1,2,2)).tolist()

            # Converting transformed keypoints from [x, y]-format to [x,y,visibility]-format by appending original visibilities to transformed coordinates of keypoints
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened): # Iterating over objects
                obj_keypoints = []
                for k_idx, kp in enumerate(obj): # Iterating over keypoints in each object
                    # kp - coordinates of keypoint
                    # keypoints_original[o_idx][k_idx][2] - original visibility of keypoint
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)

        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

        # Convert everything into a torch tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.tensor(bboxes_labels_original, dtype=torch.int64)
        # target["image_id"] = torch.tensor([idx])
        target["image_id"] = torch.tensor([idx], dtype=torch.int32)

        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
        img = F.to_tensor(img)

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.tensor(bboxes_labels_original, dtype=torch.int64)
        target_original["image_id"] = torch.tensor([idx])
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        target_original["keypoints"] = torch.as_tensor(keypoints_original, dtype=torch.float32)
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)

    def collate_fn(self, batch):
        return tuple(zip(*batch))


# # 5. Training

# In[12]:


def get_model(num_keypoints, weights_path=None):

    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   #rpn_anchor_generator=anchor_generator,
                                                                   num_classes = 3 # Background is the first class, second single and third double root
                                                                   )

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model


# In[13]:


# Set the CUDA_VISIBLE_DEVICES environment variable to the index of the GPU you want to use
gpu_index = 0  # Change this to the index of the GPU you want to use (e.g., 0, 1, 2, etc.)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

# Check if CUDA (GPU support) is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Print the selected GPU
print(f"Using GPU {gpu_index}:", torch.cuda.get_device_name(device))

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# In[14]:


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

KEYPOINTS_FOLDER_TRAIN = 'Data_Seperate_Object_Final/train' #Data_Seperate_Object_Final
KEYPOINTS_FOLDER_VAL = 'Data_Seperate_Object_Final/val' #Data_with_statics_Final
KEYPOINTS_FOLDER_TEST = 'Data_Seperate_Object_Final/test'  #Data_with_statics_Final

dataset_train = ClassDataset(KEYPOINTS_FOLDER_TRAIN, transform=None, demo=False)
dataset_val = ClassDataset(KEYPOINTS_FOLDER_VAL, transform=None, demo=False)
dataset_test = ClassDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=dataset_train.collate_fn, num_workers=2, pin_memory=True)
data_loader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, collate_fn=dataset_val.collate_fn, num_workers=2, pin_memory=True)
data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, collate_fn=dataset_test.collate_fn, num_workers=2, pin_memory=True)

model = get_model(num_keypoints = 2) #CHANGE THE NUMBER OF KEYPOINTS

#model = torch.nn.DataParallel(model)
model.to(device)

#params = [p for p in model.parameters() if p.requires_grad]
#optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.95, weight_decay=0.00005)
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

optimizer = torch.optim.Adam(params = model.parameters(), lr=0.0001, weight_decay=1e-6)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)

#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=0.6, patience=4, verbose=True)

#optimizer = torch.optim.Adam(params = model.parameters(), lr=0.000015, weight_decay=1e-6)

# In[16]:


from torchvision import transforms

def train_batch(batch, model, optim):
    imgs, targets = batch
    imgs = list(img.to(device) for img in imgs)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    # print(targets[0]['boxes'])
    optim.zero_grad()
    losses = model(imgs, targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optim.step()
    return loss, losses


def validate_batch(batch, model):
    with torch.no_grad():
        # model.eval()  # Assuming model is already in evaluation mode
        imgs, targets = batch
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        losses = model(imgs, targets)
        loss = sum(loss for loss in losses.values())
        return loss, losses

# In[ ]:

import wandb

#wandb.init(project="2024_07_23", entity="e18402")

model.train()

n_epochs = 1
final_epoch = 0

train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize with a large value
patience = 10  # Number of epochs to wait for improvement
best_model_state = None
early_stopped = False  # Flag to track if training stopped early

for epoch in range(n_epochs):
    # Training
    total_train_loss = 0.0
    for i, batch in enumerate(data_loader_train):
        N = len(data_loader_train)
        loss, losses = train_batch(batch, model, optimizer)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg, loss_keypoint = [losses[k] for k in
                                                                  ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg','loss_keypoint']]
        print(f"Epoch {epoch + 1}, Batch {i + 1}/{N}, "
              f"Train Loss: {loss.item()}, "
              f"Loc Loss: {loc_loss.item()}, "
              f"Regr Loss: {regr_loss.item()}, "
              f"Obj Loss: {loss_objectness.item()}, "
              f"RPN Box Reg Loss: {loss_rpn_box_reg.item()},"
              f"Loss Keypoint: {loss_keypoint.item()}")
        total_train_loss += loss.item()

    
    # Average train loss for the epoch
    avg_train_loss = total_train_loss / len(data_loader_train)
    train_losses.append(avg_train_loss)

    # Validation
    total_val_loss = 0.0
    for i, batch in enumerate(data_loader_val):
        N = len(data_loader_val)
        loss, losses = validate_batch(batch, model)
        loc_loss, regr_loss, loss_objectness, loss_rpn_box_reg, loss_keypoint = [losses[k] for k in
                                                                  ['loss_classifier', 'loss_box_reg',
                                                                   'loss_objectness', 'loss_rpn_box_reg','loss_keypoint']]
        print(f"Epoch {epoch + 1}, Batch {i + 1}/{N}, "
              f"Validation Loss: {loss.item()}, "
              f"Loc Loss: {loc_loss.item()}, "
              f"Regr Loss: {regr_loss.item()}, "
              f"Obj Loss: {loss_objectness.item()}, "
              f"RPN Box Reg Loss: {loss_rpn_box_reg.item()}"
              f"Loss Keypoint: {loss_keypoint.item()}")
        total_val_loss += loss.item()

    # Average validation loss for the epoch
    avg_val_loss = total_val_loss / len(data_loader_val)
    val_losses.append(avg_val_loss)
    
   # lr_scheduler.step()
    #lr_scheduler.step(avg_val_loss)

    # Log metrics to wandb
#    wandb.log({"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss})
    
    
    # Check for early stopping and save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()  # Save the best model state
        print(f"Now epoch number is {epoch}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} as validation loss did not improve.")
            early_stopped = True
            break

#wandb.finish()

# In[24]:


# Load the best model state if training did stop early
if early_stopped and best_model_state is not None:
    model.load_state_dict(best_model_state)


# In[25]:


# Save the model
save_path = 'keypointRCNN_script_report_04_09__01.pth'

# Save the model architecture
torch.save(model, save_path)


# In[16]:

from torchmetrics.detection import MeanAveragePrecision

# Initialize MeanAveragePrecision metric
map_metric = MeanAveragePrecision()

# Validation loop
model.eval()
with torch.no_grad():
    for batch in data_loader_train:
        imgs, targets = batch
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(imgs)

        # Move predicted tensors to the same device as the targets
        outputs = [{k: v.to(device) for k, v in out.items()} for out in outputs]

        # Update MeanAveragePrecision metric
        map_metric.update(outputs, targets)

# Calculate MeanAveragePrecision metric
map_metric_result_Train = map_metric.compute()

# Print or use the result as needed
print("mAP TRAIN:", map_metric_result_Train)

# In[16]:


# Initialize MeanAveragePrecision metric
map_metric = MeanAveragePrecision()

# Validation loop
model.eval()
with torch.no_grad():
    for batch in data_loader_val:
        imgs, targets = batch
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(imgs)

        # Move predicted tensors to the same device as the targets
        outputs = [{k: v.to(device) for k, v in out.items()} for out in outputs]

        # Update MeanAveragePrecision metric
        map_metric.update(outputs, targets)

# Calculate MeanAveragePrecision metric
map_metric_result = map_metric.compute()

# Print or use the result as needed
print("mAP VAL:", map_metric_result)



# Initialize MeanAveragePrecision metric
map_metric = MeanAveragePrecision()

# Validation loop
model.eval()
with torch.no_grad():
    for batch in data_loader_test:
        imgs, targets = batch
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(imgs)

        # Move predicted tensors to the same device as the targets
        outputs = [{k: v.to(device) for k, v in out.items()} for out in outputs]

        # Update MeanAveragePrecision metric
        map_metric.update(outputs, targets)

# Calculate MeanAveragePrecision metric
map_metric_result = map_metric.compute()

# Print or use the result as needed
print("mAP TEST:", map_metric_result)


# OKS ==================================================================================================


def keypoint_similarity(gt_kpts, pred_kpts, sigmas, areas):

    # epsilon to take care of div by 0 exception.
    EPSILON = torch.finfo(torch.float32).eps

    # Eucleidian dist squared:
    # d^2 = (x1 - x2)^2 + (y1 - y2)^2
    # Shape: (M, N, #kpts) --> [M, N, 17]
    dist_sq = (gt_kpts[:,None,:,0] - pred_kpts[...,0])**2 + (gt_kpts[:,None,:,1] - pred_kpts[...,1])**2

    # Boolean ground-truth visibility mask for v_i > 0. Shape: [M, #kpts] --> [M, 17]
    vis_mask = gt_kpts[..., 2].int() > 0

    # COCO assigns k = 2σ.
    k = 2*sigmas

    # Denominator in the exponent term. Shape: [M, 1, #kpts] --> [M, 1, 17]
    denom = 2 * (k**2) * (areas[:,None, None] + EPSILON)
    # Exponent term. Shape: [M, N, #kpts] --> [M, N, 17]
    exp_term = dist_sq / denom

    # Object Keypoint Similarity. Shape: (M, N)
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (vis_mask[:, None, :].sum(-1) + EPSILON)

    return oks
    

# get the mean from dict

def calculate_mean_of_dict_values(input_dict):

    # Ensure the dictionary is not empty to avoid division by zero
    if not input_dict:
        raise ValueError("Input dictionary is empty.")

    # Calculate the mean using the sum and len functions
    mean_value = sum(input_dict.values()) / len(input_dict)

    return mean_value

model.eval()
model.to(device)



oks_dic = {}

# Train loop
with torch.no_grad():
    for imgs, targets in data_loader_train:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(imgs)

        # Flatten and filter out zero entries and Convert to torch.tensor with dtype torch.float GROUND TRUTH
        # flat_tensor_gt = targets[0]["keypoints"].view(-1, 3).to(device)
        flat_tensor_gt = targets[0]["keypoints"].reshape(-1, 3).to(device)

        # # Flatten and filter out zero entries and Convert to torch.tensor with dtype torch.float PREDICTION
        main_subarrays = outputs[0]["keypoints"][:len(targets[0]["area"]) ,...]
        flat_tensor_pd = main_subarrays.reshape(-1, 3).to(device)


        NUM_KPTs = len(flat_tensor_gt)
        KPTS_OKS_SIGMAS_UNIF = torch.ones(NUM_KPTs)/NUM_KPTs

        oks_scores = keypoint_similarity(flat_tensor_gt.unsqueeze(0), flat_tensor_pd.unsqueeze(0), sigmas=KPTS_OKS_SIGMAS_UNIF.to(device), areas=targets[0]["area"].to(device))

        # Calculate the mean of the tensor
        oks_mean = torch.mean(oks_scores)

        #insert to dict
        oks_dic[targets[0]['image_id'].item()] = oks_mean.item()

OKS_train = calculate_mean_of_dict_values(oks_dic)

print("TRAIN OKS:", OKS_train)



oks_dic = {}

# Validation loop
with torch.no_grad():
    for imgs, targets in data_loader_val:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(imgs)

        # Flatten and filter out zero entries and Convert to torch.tensor with dtype torch.float GROUND TRUTH
        # flat_tensor_gt = targets[0]["keypoints"].view(-1, 3).to(device)
        flat_tensor_gt = targets[0]["keypoints"].reshape(-1, 3).to(device)

        # # Flatten and filter out zero entries and Convert to torch.tensor with dtype torch.float PREDICTION
        main_subarrays = outputs[0]["keypoints"][:len(targets[0]["area"]) ,...]
        flat_tensor_pd = main_subarrays.reshape(-1, 3).to(device)

        NUM_KPTs = len(flat_tensor_gt)
        KPTS_OKS_SIGMAS_UNIF = torch.ones(NUM_KPTs)/NUM_KPTs

        oks_scores = keypoint_similarity(flat_tensor_gt.unsqueeze(0), flat_tensor_pd.unsqueeze(0), sigmas=KPTS_OKS_SIGMAS_UNIF.to(device), areas=targets[0]["area"].to(device))

        # Calculate the mean of the tensor
        oks_mean = torch.mean(oks_scores)

        #insert to dict
        oks_dic[targets[0]['image_id'].item()] = oks_mean.item()

OKS_val = calculate_mean_of_dict_values(oks_dic)
print("VAL OKS:", OKS_val)



oks_dic = {}

# Testing loop
with torch.no_grad():
    for imgs, targets in data_loader_test:
        imgs = list(img.to(device) for img in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        outputs = model(imgs)

        # Flatten and filter out zero entries and Convert to torch.tensor with dtype torch.float GROUND TRUTH
        # flat_tensor_gt = targets[0]["keypoints"].view(-1, 3).to(device)
        flat_tensor_gt = targets[0]["keypoints"].reshape(-1, 3).to(device)

        # # Flatten and filter out zero entries and Convert to torch.tensor with dtype torch.float PREDICTION
        main_subarrays = outputs[0]["keypoints"][:len(targets[0]["area"]) ,...]
        flat_tensor_pd = main_subarrays.reshape(-1, 3).to(device)


        NUM_KPTs = len(flat_tensor_gt)
        KPTS_OKS_SIGMAS_UNIF = torch.ones(NUM_KPTs)/NUM_KPTs

        oks_scores = keypoint_similarity(flat_tensor_gt.unsqueeze(0), flat_tensor_pd.unsqueeze(0), sigmas=KPTS_OKS_SIGMAS_UNIF.to(device), areas=targets[0]["area"].to(device))

        # Calculate the mean of the tensor
        oks_mean = torch.mean(oks_scores)

        #insert to dict
        oks_dic[targets[0]['image_id'].item()] = oks_mean.item()

OKS_test = calculate_mean_of_dict_values(oks_dic)

print("TEST OKS:", OKS_test)



import datetime

# Define the path to the log file
log_file_path = "log_file.txt"

# Open the log file in append mode
with open(log_file_path, "a") as log_file:
    # Get the current date and time
    current_time = datetime.datetime.now()

    # Write a log message with the current date and time
    log_message = f"{current_time}: Script executed successfully\n"
   # log_file.write(log_message)
   # log_file.write(map_metric_result_Train)
   # log_file.write(map_metric_result)
   # log_file.write(OKS_train)
   # log_file.write(OKS_test)


# For example, print a message to the console
print("Script executed successfully")



