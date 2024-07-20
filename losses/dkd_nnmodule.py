import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupleMLLoss(nn.Module):
    def __init__(self, alpha, beta, temperature):
        super(DecoupleMLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, logits1, logits2, target):
        gt_mask = _get_gt_mask(logits1, target)
        other_mask = _get_other_mask(logits1, target)
        pred_net1 = F.softmax(logits1 / self.temperature, dim=1)
        pred_net2 = F.softmax(logits2 / self.temperature, dim=1)
        pred_net1 = cat_mask(pred_net1, gt_mask, other_mask)
        pred_net2 = cat_mask(pred_net2, gt_mask, other_mask)
        log_pred_net1 = torch.log(pred_net1)
        tckd_loss = (
            F.kl_div(log_pred_net1, pred_net2, reduction='batchmean')
            * (self.temperature ** 2)
        )
        pred_net2_part2 = F.softmax(
            logits2 / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_net1_part2 = F.log_softmax(
            logits1 / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_net1_part2, pred_net2_part2, reduction='batchmean')
            * (self.temperature ** 2)
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Decouple_ML(nn.Module):
#     """Decoupled mutual learning"""
#
#     def __init__(self, nets, cfg):
#         super(Decouple_ML, self).__init__()
#         self.nets = nets
#         self.number_net = len(nets)
#         self.alpha = cfg.alpha
#         self.beta = cfg.beta
#         self.temperature = cfg.kd_T
#         self.warmup = cfg.warmup
#
#     def train(self, mode=True):
#         if not isinstance(mode, bool):
#             raise ValueError("training mode is expected to be boolean")
#         self.training = mode
#         for net in self.nets:
#             net.train(mode)
#         return self
#
#     def get_learnable_parameters(self):
#         params = []
#         for net in self.nets:
#             params.extend([v for k, v in net.named_parameters()])
#         return params
#
#     def forward(self, image, target, epoch, **kwargs):
#         logits_nets = []
#         for net in self.nets:
#             logits, _ = net(image)
#             logits_nets.append(logits)
#
#         # Calculate decouple_ml_loss
#         total_loss = 0
#         for i in range(self.number_net):
#             for j in range(i + 1, self.number_net):
#                 total_loss += min(epoch / self.warmup, 1.0) * decouple_ml_loss(
#                     logits_nets[i],
#                     logits_nets[j],
#                     target,
#                     self.alpha,
#                     self.beta,
#                     self.temperature,
#                 )
#         return {"loss_decouple_ml": total_loss}
#
#
# def decouple_ml_loss(logits_net1, logits_net2, target, alpha, beta, temperature):
#     gt_mask = _get_gt_mask(logits_net1, target)
#     other_mask = _get_other_mask(logits_net1, target)
#     pred_net1 = F.softmax(logits_net1 / temperature, dim=1)
#     pred_net2 = F.softmax(logits_net2 / temperature, dim=1)
#     pred_net1 = cat_mask(pred_net1, gt_mask, other_mask)
#     pred_net2 = cat_mask(pred_net2, gt_mask, other_mask)
#     log_pred_net1 = torch.log(pred_net1)
#     tckd_loss = (
#         F.kl_div(log_pred_net1, pred_net2, reduction='batchmean')
#         * (temperature ** 2)
#     )
#     pred_net2_part2 = F.softmax(
#         logits_net2 / temperature - 1000.0 * gt_mask, dim=1
#     )
#     log_pred_net1_part2 = F.log_softmax(
#         logits_net1 / temperature - 1000.0 * gt_mask, dim=1
#     )
#     nckd_loss = (
#         F.kl_div(log_pred_net1_part2, pred_net2_part2, reduction='batchmean')
#         * (temperature ** 2)
#     )
#     return alpha * tckd_loss + beta * nckd_loss
#
#
# def _get_gt_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
#     return mask
#
#
# def _get_other_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
#     return mask
#
#
# def cat_mask(t, mask1, mask2):
#     t1 = (t * mask1).sum(dim=1, keepdims=True)
#     t2 = (t * mask2).sum(1, keepdims=True)
#     rt = torch.cat([t1, t2], dim=1)
#     return rt
