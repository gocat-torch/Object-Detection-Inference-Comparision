import torch.nn as nn
import torch.nn.functional as F
import torch


from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device, reduction = 'mean'):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)
        self.reduction = reduction

    def forward(self, confidence, predicted_locations, labels, gt_locations ):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        reduction = self.reduction
        if reduction == 'none':
            return self.forward_dp(confidence, predicted_locations, labels, gt_locations )
        
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        # print(confidence.shape , 'confidence.shape prior')

        confidence = confidence[mask, :]
        if reduction == 'none':
            print('do reduction')
            print(confidence.shape , 'confidence.shape')
            print(mask.shape , 'mask.shape')

            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False, reduction='none')
        else:
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        if reduction == 'none':
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False, reduction='none')
        else:
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        
        # print(classification_loss)
        return smooth_l1_loss/num_pos, classification_loss/num_pos
    def forward_dp(self, confidence, predicted_locations, labels, gt_locations ):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        reduction = self.reduction
        num_classes = confidence.size(2)
        batch_size  = confidence.size(0)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        # print(confidence.shape , 'confidence.shape prior')

        # confidence = confidence[mask, :]
        if reduction == 'none':
            # print('do reduction')
            # print(confidence.shape , 'confidence.shape')
            # print(mask.shape , 'mask.shape')

            classification_loss = [ F.cross_entropy(confidence[b][mask[b],:].reshape(-1, num_classes), labels[b][mask[b]], size_average=False) for b in range(batch_size) ] 
            # classification_loss = torch.Tensor(classification_loss)
            # print(classification_loss.shape , 'classification_loss.shape')

        else:
            classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        # predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        # gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        if reduction == 'none':
            # print(pos_mask.shape , 'pos_mask.shape')
            # print(predicted_locations.shape , 'predicted_locations.shape')
            # print(gt_locations.shape , 'gt_locations.shape')

            smooth_l1_loss = [ F.smooth_l1_loss(predicted_locations[b][pos_mask[b], :], gt_locations[b][pos_mask[b], :], size_average=False)  for b in range(batch_size) ] 
            # smooth_l1_loss = torch.cat(smooth_l1_loss)

            # print(smooth_l1_loss.shape , 'smooth_l1_loss.shape')

        else:
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        
        # print(classification_loss)
        classification_loss = [ item /num_pos for item in classification_loss]
        smooth_l1_loss      = [ item /num_pos for item in smooth_l1_loss]

        return smooth_l1_loss, classification_loss
