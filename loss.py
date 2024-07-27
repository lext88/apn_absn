from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class CrossEntropyLossWithAttention(nn.Module):
    def __init__(self, alpha=0.5):
        super(CrossEntropyLossWithAttention, self).__init__()
        self.alpha = alpha
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, attention_weights=None):
        input_ = inputs.view(inputs.size(0), inputs.size(1), -1)

        log_probs = self.logsoftmax(input_)
        targets_ = torch.zeros(input_.size(0), input_.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets_ = targets_.unsqueeze(-1)
        targets_ = targets_.cuda()

        loss = (- targets_ * log_probs).mean(0).sum()
        loss = loss / input_.size(2)
        
        if attention_weights is not None:
            loss = loss * attention_weights.mean()
        
        return loss

class SupCluLossWithAttention(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupCluLossWithAttention, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, attention_weights=None):
        device = torch.device('cuda' if features.is_cuda else 'cpu')

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = 1
        contrast_feature = features
        if self.contrast_mode == 'one':
            assert False
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        if attention_weights is not None:
            loss = loss * attention_weights.mean()

        return loss

class AdaptivePrototypicalLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(AdaptivePrototypicalLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLossWithAttention(alpha=alpha)
        self.sup_clu_loss = SupCluLossWithAttention(temperature=temperature, contrast_mode=contrast_mode, base_temperature=base_temperature)

    def forward(self, inputs, targets, features, labels=None, mask=None, attention_weights=None):
        ce_loss = self.cross_entropy_loss(inputs, targets, attention_weights)
        sup_clu_loss = self.sup_clu_loss(features, labels, mask, attention_weights)

        # Combine losses with a weighted sum
        total_loss = ce_loss + sup_clu_loss

        return total_loss
