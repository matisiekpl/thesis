import torch


def intersection_over_union(boxes_predictions, boxes_labels):
    box1_x1 = boxes_predictions[..., 0:1] - boxes_predictions[..., 2:3] / 2
    box1_y1 = boxes_predictions[..., 1:2] - boxes_predictions[..., 3:4] / 2
    box1_x2 = boxes_predictions[..., 0:1] + boxes_predictions[..., 2:3] / 2
    box1_y2 = boxes_predictions[..., 1:2] + boxes_predictions[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
