import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):

    def __init__(self, S=7, B=2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 1:5], target[..., 1:5])
        iou_b2 = intersection_over_union(predictions[..., 6:10], target[..., 1:5])
        iou = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, best_box_index = torch.max(iou, dim=0)
        does_box_exists = target[..., 0].unsqueeze(3)

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        box_predictions = does_box_exists * ((
                best_box_index * predictions[..., 6:10]
                + (1 - best_box_index) * predictions[..., 1:5]
        ))

        box_targets = does_box_exists * target[..., 1:5]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
                does_box_exists * predictions[..., 5:6] + (1 - does_box_exists) * predictions[..., 0:1]
        )

        object_loss = self.mse(
            torch.flatten(does_box_exists * pred_box),
            torch.flatten(does_box_exists * target[..., 0:1]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        # no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )

        no_object_loss = self.mse(
            torch.flatten((1 - does_box_exists) * predictions[..., 0:1], start_dim=1),
            torch.flatten((1 - does_box_exists) * target[..., 0:1], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - does_box_exists) * predictions[..., 5:6], start_dim=1),
            torch.flatten((1 - does_box_exists) * target[..., 0:1], start_dim=1)
        )

        loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
        )

        return loss
