# ==== Data Augmentation Transforms ====

def train_transform():
    """
    Compose training augmentations using Albumentations.
    Currently applies only CLAHE for local contrast enhancement.
    You can enable other augmentations as needed.
    """
    return A.Compose([
        A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
    )

def val_transform():
    """
    Compose validation augmentations.
    Here, just CLAHE for consistency. No random augmentations applied.
    """
    return A.Compose([
        A.CLAHE(clip_limit=40.0, tile_grid_size=(8, 8), always_apply=True, p=1.0),
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
    )
