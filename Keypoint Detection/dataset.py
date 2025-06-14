# ==== Custom Dataset for Keypoint Detection ====

class ClassDataset(Dataset):
    """
    PyTorch Dataset for loading keypoint and bounding box data from a folder.

    Args:
        root (str): Path to data root directory with 'images' and 'annotations' subfolders.
        transform (callable, optional): Albumentations transform pipeline.
        demo (bool): If True, returns both original and transformed images/targets (useful for visualization).
    """
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))

    def __getitem__(self, idx):
        # Load image and annotation file
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations_files[idx])
        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        with open(ann_path) as f:
            data = json.load(f)
        bboxes_original = data['bboxes']
        keypoints_original = [[subsub for subsub in sublist[4:]] for sublist in data['keypoints']]  # This is just an example (only APEX) and please take these according to your requirements 
        keypoints_original = [sorted(sublist, key=lambda x: x[0]) for sublist in keypoints_original]
        bboxes_labels_original = data['labels']

        if self.transform:
            # Flatten keypoints for Albumentations
            keypoints_original_flattened = [el[0:2] for kp in keypoints_original for el in kp]
            transformed = self.transform(
                image=img_original,
                bboxes=bboxes_original,
                bboxes_labels=bboxes_labels_original,
                keypoints=keypoints_original_flattened,
            )
            img = transformed['image']
            bboxes = transformed['bboxes']
            # Unflatten keypoints back to [num_objects, num_keypoints, 2]
            keypoints_transformed_unflattened = np.reshape(np.array(transformed['keypoints']), (-1, 2, 2)).tolist()
            # Restore [x, y, visibility]
            keypoints = []
            for o_idx, obj in enumerate(keypoints_transformed_unflattened):
                obj_keypoints = []
                for k_idx, kp in enumerate(obj):
                    obj_keypoints.append(kp + [keypoints_original[o_idx][k_idx][2]])
                keypoints.append(obj_keypoints)
        else:
            img, bboxes, keypoints = img_original, bboxes_original, keypoints_original

        # Convert all fields to torch tensors
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {
            "boxes": bboxes,
            "labels": torch.tensor(bboxes_labels_original, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int32),
            "area": (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0]),
            "iscrowd": torch.zeros(len(bboxes), dtype=torch.int64),
            "keypoints": torch.as_tensor(keypoints, dtype=torch.float32),
        }
        img = F.to_tensor(img)

        if self.demo:
            # Also return original image/target (useful for debugging/visualization)
            bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
            target_original = {
                "boxes": bboxes_original,
                "labels": torch.tensor(bboxes_labels_original, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0]),
                "iscrowd": torch.zeros(len(bboxes_original), dtype=torch.int64),
                "keypoints": torch.as_tensor(keypoints_original, dtype=torch.float32),
            }
            img_original = F.to_tensor(img_original)
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)

    def collate_fn(self, batch):
        # Custom collate for variable-sized targets
        return tuple(zip(*batch))
