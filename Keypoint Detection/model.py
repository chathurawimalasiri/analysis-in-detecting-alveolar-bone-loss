# ==== Model Definition ====

def get_model(num_keypoints, weights_path=None):
    """
    Create a Keypoint R-CNN model for keypoint detection.

    Args:
        num_keypoints (int): Number of keypoints per object.
        weights_path (str, optional): Path to pre-trained weights.

    Returns:
        torch.nn.Module: The initialized Keypoint R-CNN model.
    """
    anchor_generator = AnchorGenerator(
        sizes=(32, 64, 128, 256, 512),
        aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0)
    )
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=True,
        num_keypoints=num_keypoints,
        num_classes=3,  # e.g., 1 background + 2 object types
        # rpn_anchor_generator=anchor_generator, # Uncomment if custom anchors needed
    )
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
    return model
