import torch
from torch import nn
import math
import torch.nn.functional as F


class DecoupleMCL(nn.Module):
    def __init__(self, args):
        super(DecoupleMCL, self).__init__()
        self.number_net = args.number_net
        # self.feat_dim = args.feat_dim
        self.args = args
        # self.kl = KLDiv(T=args.kd_T)

    def forward(self, embeddings, labels):
        batchSize = embeddings[0].size(0)

        labels = labels.unsqueeze(0)






        return loss_dmcl


class DMCL_Loss(nn.Module):
    def __init__(self, args):
        super(DMCL_Loss, self).__init__()
        self.embed_list = nn.ModuleList([])
        self.args = args
        for i in range(args.number_net):
            self.embed_list.append(Embed(args.rep_dim[i], args.feat_dim))

        self.contrast = SupMCL(args)

    def forward(self, embeddings, labels):



        return vcl_loss, soft_vcl_loss


class Embed(nn.Module):
    """Embedding module"""

    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.proj_head(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class decople_KLDiv(nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T, ):
        super(decople_KLDiv, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='batchmean') * (self.T ** 2)
        return loss








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









