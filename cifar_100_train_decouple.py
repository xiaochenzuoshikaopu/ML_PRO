

# v1
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, correct_num
from tqdm import tqdm

from bisect import bisect_right
import time
import math
from losses.dkd_nnmodule import DecoupleMLLoss
from dataset.class_sampler import MPerClassSampler

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet32', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--lr-type', default='cosine', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 225], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str, help='evaluate checkpoint')
parser.add_argument('--number-net', type=int, default=2, help='number of networks')
parser.add_argument('--logit-distill', action='store_true', help='combine with decouple logit distillation')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
parser.add_argument('--resume-lr', type=float, help='learning rate to use on resume')  # 恢复训练时的学习率

parser.add_argument('--kd_T', type=float, default=4.0, help='temperature of KL-divergence')
parser.add_argument('--tau', default=0.1, type=float, help='temperature for contrastive distribution')
parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for VCL')
parser.add_argument('--beta', type=float, default=8.0, help='weight balance for ICL')
parser.add_argument('--warmup', type=int, default=20, help='number of warmup epochs')
parser.add_argument('--feat-dim', default=128, type=int, help='feature dimension')

# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.isdir('./result/'):
    os.makedirs('./result/')

log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + \
          'arch' + '_' + args.arch + '_' + \
          'dataset' + '_' + args.dataset + '_' + \
          'seed' + str(args.manual_seed) + '.txt'

with open(log_txt, 'a+') as f:
    f.write("==========\nArgs:{}\n==========".format(args) + '\n')

np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                  [0.2675, 0.2565, 0.2761])
                                         ]))

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                 [0.2675, 0.2565, 0.2761]),
                                        ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          sampler=MPerClassSampler(labels=trainset.targets, m=2,
                                                                   batch_size=args.batch_size,
                                                                   length_before_new_iter=len(trainset.targets)),
                                          pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                         pin_memory=(torch.cuda.is_available()))
# --------------------------------------------------------------------------------------------

# Model
print('==> Building model..')
model = getattr(models, args.arch)
net = model(num_classes=num_classes, number_net=args.number_net)
net.eval()
resolution = (1, 3, 32, 32)
print('Arch: %s, Params: %.2fM, FLOPs: %.2fG'
      % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, resolution) / 1e9))
del (net)

net = model(num_classes=num_classes, number_net=args.number_net).cuda()
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

class DecoupleMLLoss(nn.Module):
    def __init__(self, alpha, beta, temperature):
        super(DecoupleMLLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, logits1, logits2, target):
        gt_mask = self._get_gt_mask(logits1, target)
        other_mask = self._get_other_mask(logits1, target)
        pred_net1 = F.softmax(logits1 / self.temperature, dim=1)
        pred_net2 = F.softmax(logits2 / self.temperature, dim=1)
        pred_net1 = self.cat_mask(pred_net1, gt_mask, other_mask)
        pred_net2 = self.cat_mask(pred_net2, gt_mask, other_mask)
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

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

def train(epoch, criterion_list, optimizer):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
    train_loss_decouple_ml = AverageMeter('train_loss_decouple_ml', ':.4e')

    top1_num = [0] * args.number_net
    top5_num = [0] * args.number_net
    total = [0] * args.number_net

    lr = adjust_lr(optimizer, epoch, args)
    print(f'Epoch {epoch}, Learning Rate: {lr}')

    start_time = time.time()
    criterion_ce = criterion_list[0]
    criterion_decouple_ml = criterion_list[1]

    net.train()
    with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", ncols=100) as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.float().cuda()
            targets = targets.cuda()

            # optimizer_ce.zero_grad()
            # optimizer_decouple.zero_grad()
            optimizer.zero_grad()
            # 计算logits
            logits, embeddings = net(inputs)
            # 计算分类损失
            for i in range(len(logits)):
                loss_cls = loss_cls + criterion_ce(logits[i], targets)

            # 集成logits
            ensemble_logits = 0.
            for i in range(len(logits)):
                ensemble_logits = ensemble_logits + logits[i]
            ensemble_logits = ensemble_logits / len(logits)
            ensemble_logits = ensemble_logits.detach()

            # 如果启用logit蒸馏
            if args.logit_distill:
                for i in range(len(logits)):
                    loss_decouple_kd = loss_decouple_kd + criterion_decouple_ml(logits[i], ensemble_logits, targets)

                # 如果有多个网络，进行解耦蒸馏
                for i in range(args.number_net):
                    for j in range(args.number_net):
                        if i != j:
                            loss_decouple_kd = loss_decouple_kd + criterion_decouple_ml(logits[i], logits[j], targets)

            loss = loss_cls + loss_decouple_kd
            # loss_cls.backward(retain_graph=True)
            # optimizer_ce.step()

            # loss_decouple_kd.backward()
            # optimizer_decouple.step()
            loss.backward()
            optimizer.step()
            # 分别计算梯度并更新
            # loss_cls.backward(retain_graph=True)
            # optimizer_ce.step()

            # loss_decouple_kd.backward()
            # optimizer_decouple.step()

            train_loss.update(loss.item(), inputs.size(0))
            train_loss_cls.update(loss_cls.item(), inputs.size(0))
            train_loss_decouple_ml.update(loss_decouple_kd.item(), inputs.size(0))

            for i in range(len(logits)):
                top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
                top1_num[i] += top1
                top5_num[i] += top5
                total[i] += targets.size(0)

            pbar.set_postfix({
                "loss": train_loss.avg,
                "cls_loss": train_loss_cls.avg,
                "decouple_loss": train_loss_decouple_ml.avg,
                "Top-1 Acc": (top1_num[0] / total[0]).item() if total[0] != 0 else 0,
                "Top-5 Acc": (top5_num[0] / total[0]).item() if total[0] != 0 else 0,
                "eta": pbar.format_dict["elapsed"] * (len(pbar) / (pbar.n + 1)) - pbar.format_dict["elapsed"]
            })
            pbar.update(1)

    acc1 = [round((top1_num[i] / total[i]).item(), 4) for i in range(args.number_net)]
    acc5 = [round((top5_num[i] / total[i]).item(), 4) for i in range(args.number_net)]

    with open(log_txt, 'a+') as f:
        f.write('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                '\n Train_loss:{:.5f}'
                '\t Train_loss_cls:{:.5f}'
                '\t Train_loss_decouple_ml:{:.5f}'
                '\n Train top-1 accuracy: {} \n'
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg,
                        train_loss_cls.avg,
                        train_loss_decouple_ml.avg,
                        str(acc1)))

    print(f'Epoch {epoch} Completed, Total Loss: {train_loss.avg}, Classification Loss: {train_loss_cls.avg}, Decouple Loss: {train_loss_decouple_ml.avg}')

def test(epoch, criterion_ce):
    net.eval()
    global best_acc
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    top1_num = [0] * (args.number_net + 1)
    top5_num = [0] * (args.number_net + 1)
    total = [0] * (args.number_net + 1)

    with torch.no_grad():
        with tqdm(total=len(testloader), desc=f"Test Epoch {epoch + 1}/{args.epochs}", unit="batch", ncols=100) as pbar:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                logits, embeddings = net(inputs)

                loss_cls = 0.
                ensemble_logits = 0

                for i in range(len(logits)):
                    loss_cls = loss_cls + criterion_ce(logits[i], targets)
                for i in range(len(logits)):
                    ensemble_logits = ensemble_logits + logits[i]

                test_loss_cls.update(loss_cls, inputs.size(0))

                for i in range(args.number_net + 1):
                    if i == args.number_net:
                        top1, top5 = correct_num(ensemble_logits, targets, topk=(1, 5))
                    else:
                        top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
                    top1_num[i] += top1
                    top5_num[i] += top5
                    total[i] += targets.size(0)

                pbar.set_postfix({
                    "Test Loss": test_loss_cls.avg,
                    "Net1 Top-1 Acc": (top1_num[0] / total[0]).item(),
                    "Net2 Top-1 Acc": (top1_num[1] / total[1]).item(),
                    "Ensemble Top-1 Acc": (top1_num[2] / total[2]).item(),
                    "Net1 Top-5 Acc": (top5_num[0] / total[0]).item(),
                    "Net2 Top-5 Acc": (top5_num[1] / total[1]).item(),
                    "Ensemble Top-5 Acc": (top5_num[2] / total[2]).item(),
                    "eta": pbar.format_dict["elapsed"] * (len(pbar) / (pbar.n + 1)) - pbar.format_dict["elapsed"]
                })
                pbar.update(1)

        acc1 = [round((top1_num[i] / total[i]).item(), 4) for i in range(args.number_net + 1)]
        acc5 = [round((top5_num[i] / total[i]).item(), 4) for i in range(args.number_net + 1)]

        with open(log_txt, 'a+') as f:
            f.write('Test epoch:{}\t Test_loss_cls:{:.5f}\t Test top-1 accuracy:{}\t Test top-5 accuracy:{}\n'
                    .format(epoch, test_loss_cls.avg, str(acc1), str(acc5)))

        print('Test epoch:{}\t Test top-1 accuracy:{}\t Test top-5 accuracy:{}\n'.format(epoch, str(acc1), str(acc5)))

    return max(acc1[:-1])

if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_ce = nn.CrossEntropyLoss()
    criterion_decouple_ml = DecoupleMLLoss(args.alpha, args.beta, args.kd_T)

    if args.evaluate:
        print('Evaluate pre-trained weights from: {}'.format(args.evaluate_checkpoint))
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_ce)
    else:
        # optimizer_ce = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=args.weight_decay,
        #                          nesterov=True)
        # optimizer_decouple = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=args.weight_decay,
                                       # nesterov=True)

        # criterion_list = [criterion_ce, criterion_decouple_ml]
        optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        criterion_list = [criterion_ce, criterion_decouple_ml]

        if args.resume:
            print('Resume pre-trained weights from: {}'.format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            net.module.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1
            # 如果指定了 resume-lr 参数，则重新设置学习率
            if args.resume_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.resume_lr
                print(f"Learning rate reset to: {args.resume_lr}")

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_ce)

            state = {
                'net': net.module.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(
            os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_ce)

        with open(log_txt, 'a+') as f:
            f.write('Test top-1 best_accuracy: {} \n'.format(top1_acc))
        print('Test top-1 best_accuracy: {} \n'.format(top1_acc))


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F
#
# import os
# import shutil
# import argparse
# import numpy as np
#
# import models
# import torchvision
# import torchvision.transforms as transforms
# from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, correct_num
# from tqdm import tqdm
#
# from bisect import bisect_right
# import time
# import math
# from losses.dkd_nnmodule import DecoupleMLLoss
# from dataset.class_sampler import MPerClassSampler
#
# parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
# parser.add_argument('--data', default='./data/', type=str, help='Dataset directory')
# parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
# parser.add_argument('--arch', default='resnet32', type=str, help='network architecture')
# parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
# parser.add_argument('--lr-type', default='cosine', type=str, help='learning rate strategy')
# parser.add_argument('--milestones', default=[150, 225], type=int, nargs='+', help='milestones for lr-multistep')
# parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
# parser.add_argument('--batch-size', type=int, default=128, help='batch size')
# parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
# parser.add_argument('--gpu-id', type=str, default='0')
# parser.add_argument('--manual_seed', type=int, default=0)
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
# parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
# parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str, help='evaluate checkpoint')
# parser.add_argument('--number-net', type=int, default=2, help='number of networks')
# parser.add_argument('--logit-distill', action='store_true', help='combine with decouple logit distillation')
# parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
#
# parser.add_argument('--kd_T', type=float, default=4.0, help='temperature of KL-divergence')
# parser.add_argument('--tau', default=0.1, type=float, help='temperature for contrastive distribution')
# parser.add_argument('--alpha', type=float, default=1.0, help='weight balance for VCL')
# parser.add_argument('--beta', type=float, default=8.0, help='weight balance for ICL')
# parser.add_argument('--warmup', type=int, default=20, help='number of warmup epochs')
# parser.add_argument('--feat-dim', default=128, type=int, help='feature dimension')
#
# # global hyperparameter set
# args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
#
# if not os.path.isdir(args.checkpoint_dir):
#     os.makedirs(args.checkpoint_dir)
# if not os.path.isdir('./result/'):
#     os.makedirs('./result/')
#
# log_txt = 'result/' + str(os.path.basename(__file__).split('.')[0]) + '_' + \
#           'arch' + '_' + args.arch + '_' + \
#           'dataset' + '_' + args.dataset + '_' + \
#           'seed' + str(args.manual_seed) + '.txt'
#
# with open(log_txt, 'a+') as f:
#     f.write("==========\nArgs:{}\n==========".format(args) + '\n')
#
# np.random.seed(args.manual_seed)
# torch.manual_seed(args.manual_seed)
# torch.cuda.manual_seed_all(args.manual_seed)
#
# num_classes = 100
# trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
#                                          transform=transforms.Compose([
#                                              transforms.RandomCrop(32, padding=4),
#                                              transforms.RandomHorizontalFlip(),
#                                              transforms.ToTensor(),
#                                              transforms.Normalize([0.5071, 0.4867, 0.4408],
#                                                                   [0.2675, 0.2565, 0.2761])
#                                          ]))
#
# testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
#                                         transform=transforms.Compose([
#                                             transforms.ToTensor(),
#                                             transforms.Normalize([0.5071, 0.4867, 0.4408],
#                                                                  [0.2675, 0.2565, 0.2761]),
#                                         ]))
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
#                                           sampler=MPerClassSampler(labels=trainset.targets, m=2,
#                                                                    batch_size=args.batch_size,
#                                                                    length_before_new_iter=len(trainset.targets)),
#                                           pin_memory=(torch.cuda.is_available()))
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
#                                          pin_memory=(torch.cuda.is_available()))
# # --------------------------------------------------------------------------------------------
#
# # Model
# print('==> Building model..')
# model = getattr(models, args.arch)
# net = model(num_classes=num_classes, number_net=args.number_net)
# net.eval()
# resolution = (1, 3, 32, 32)
# print('Arch: %s, Params: %.2fM, FLOPs: %.2fG'
#       % (args.arch, cal_param_size(net) / 1e6, cal_multi_adds(net, resolution) / 1e9))
# del (net)
#
# net = model(num_classes=num_classes, number_net=args.number_net).cuda()
# net = torch.nn.DataParallel(net)
# cudnn.benchmark = True
#
# class DecoupleMLLoss(nn.Module):
#     def __init__(self, alpha, beta, temperature):
#         super(DecoupleMLLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.temperature = temperature
#
#     def forward(self, logits1, logits2, target):
#         gt_mask = _get_gt_mask(logits1, target)
#         other_mask = _get_other_mask(logits1, target)
#         pred_net1 = F.softmax(logits1 / self.temperature, dim=1)
#         pred_net2 = F.softmax(logits2 / self.temperature, dim=1)
#         pred_net1 = cat_mask(pred_net1, gt_mask, other_mask)
#         pred_net2 = cat_mask(pred_net2, gt_mask, other_mask)
#         log_pred_net1 = torch.log(pred_net1)
#         tckd_loss = (
#             F.kl_div(log_pred_net1, pred_net2, reduction='batchmean')
#             * (self.temperature ** 2)
#         )
#         pred_net2_part2 = F.softmax(
#             logits2 / self.temperature - 1000.0 * gt_mask, dim=1
#         )
#         log_pred_net1_part2 = F.log_softmax(
#             logits1 / self.temperature - 1000.0 * gt_mask, dim=1
#         )
#         nckd_loss = (
#             F.kl_div(log_pred_net1_part2, pred_net2_part2, reduction='batchmean')
#             * (self.temperature ** 2)
#         )
#         return self.alpha * tckd_loss + self.beta * nckd_loss
#
# def _get_gt_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
#     return mask
#
# def _get_other_mask(logits, target):
#     target = target.reshape(-1)
#     mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
#     return mask
#
# def cat_mask(t, mask1, mask2):
#     t1 = (t * mask1).sum(dim=1, keepdims=True)
#     t2 = (t * mask2).sum(1, keepdims=True)
#     rt = torch.cat([t1, t2], dim=1)
#     return rt
#
# def train(epoch, criterion_list, optimizer):
#     train_loss = AverageMeter('train_loss', ':.4e')
#     train_loss_cls = AverageMeter('train_loss_cls', ':.4e')
#     train_loss_decouple_ml = AverageMeter('train_loss_decouple_ml', ':.4e')
#
#     top1_num = [0] * args.number_net
#     top5_num = [0] * args.number_net
#     total = [0] * args.number_net
#
#     lr = adjust_lr(optimizer, epoch, args)
#
#     start_time = time.time()
#     criterion_ce = criterion_list[0]
#     criterion_decouple_ml = criterion_list[1]
#
#     net.train()
#     with tqdm(total=len(trainloader), desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch", ncols=100) as pbar:
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             inputs = inputs.float().cuda()
#             targets = targets.cuda()
#
#             optimizer.zero_grad()
#             logits, embeddings = net(inputs)
#
#             loss_cls = torch.tensor(0.).cuda()
#             loss_decouple_kd = torch.tensor(0.).cuda()
#
#             for i in range(len(logits)):
#                 loss_cls = loss_cls + criterion_ce(logits[i], targets)
#
#             ensemble_logits = 0.
#             for i in range(len(logits)):
#                 ensemble_logits = ensemble_logits + logits[i]
#             ensemble_logits = ensemble_logits / len(logits)
#             ensemble_logits = ensemble_logits.detach()
#
#             if args.logit_distill:
#                 if args.number_net == 2:
#                     loss_decouple_kd = loss_decouple_kd + criterion_decouple_ml(logits[0], logits[1], targets)
#                     loss_decouple_kd = loss_decouple_kd + criterion_decouple_ml(logits[1], logits[0], targets)
#                 else:
#                     for i in range(args.number_net):
#                         for j in range(args.number_net):
#                             if i != j:
#                                 loss_decouple_kd = loss_decouple_kd + criterion_decouple_ml(logits[i], logits[j], targets)
#
#             loss = loss_cls + loss_decouple_kd
#
#             loss.backward()
#             optimizer.step()
#
#             train_loss.update(loss.item(), inputs.size(0))
#             train_loss_cls.update(loss_cls.item(), inputs.size(0))
#             train_loss_decouple_ml.update(loss_decouple_kd.item(), inputs.size(0))
#
#             for i in range(len(logits)):
#                 top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
#                 top1_num[i] += top1
#                 top5_num[i] += top5
#                 total[i] += targets.size(0)
#
#             pbar.set_postfix({
#                 "loss": train_loss.avg,
#                 "Top-1 Acc": (top1_num[0] / total[0]).item(),
#                 "Top-5 Acc": (top5_num[0] / total[0]).item(),
#                 "eta": pbar.format_dict["elapsed"] * (len(pbar) / (pbar.n + 1)) - pbar.format_dict["elapsed"]
#             })
#             pbar.update(1)
#
#     acc1 = [round((top1_num[i] / total[i]).item(), 4) for i in range(args.number_net)]
#     acc5 = [round((top5_num[i] / total[i]).item(), 4) for i in range(args.number_net)]
#
#     with open(log_txt, 'a+') as f:
#         f.write('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
#                 '\n Train_loss:{:.5f}'
#                 '\t Train_loss_cls:{:.5f}'
#                 '\t Train_loss_decouple_ml:{:.5f}'
#                 '\n Train top-1 accuracy: {} \n'
#                 .format(epoch, lr, time.time() - start_time,
#                         train_loss.avg,
#                         train_loss_cls.avg,
#                         train_loss_decouple_ml.avg,
#                         str(acc1)))
#
#
# def test(epoch, criterion_ce):
#     net.eval()
#     global best_acc
#     test_loss_cls = AverageMeter('test_loss_cls', ':.4e')
#
#     top1_num = [0] * (args.number_net + 1)
#     top5_num = [0] * (args.number_net + 1)
#     total = [0] * (args.number_net + 1)
#
#     with torch.no_grad():
#         with tqdm(total=len(testloader), desc=f"Test Epoch {epoch + 1}/{args.epochs}", unit="batch", ncols=100) as pbar:
#             for batch_idx, (inputs, targets) in enumerate(testloader):
#                 inputs, targets = inputs.cuda(), targets.cuda()
#                 logits, embeddings = net(inputs)
#
#                 loss_cls = 0.
#                 ensemble_logits = 0
#
#                 for i in range(len(logits)):
#                     loss_cls = loss_cls + criterion_ce(logits[i], targets)
#                 for i in range(len(logits)):
#                     ensemble_logits = ensemble_logits + logits[i]
#
#                 test_loss_cls.update(loss_cls, inputs.size(0))
#
#                 for i in range(args.number_net + 1):
#                     if i == args.number_net:
#                         top1, top5 = correct_num(ensemble_logits, targets, topk=(1, 5))
#                     else:
#                         top1, top5 = correct_num(logits[i], targets, topk=(1, 5))
#                     top1_num[i] += top1
#                     top5_num[i] += top5
#                     total[i] += targets.size(0)
#
#                 pbar.set_postfix({
#                     "Test Loss": test_loss_cls.avg,
#                     "Net1 Top-1 Acc": (top1_num[0] / total[0]).item(),
#                     "Net2 Top-1 Acc": (top1_num[1] / total[1]).item(),
#                     "Ensemble Top-1 Acc": (top1_num[2] / total[2]).item(),
#                     "Net1 Top-5 Acc": (top5_num[0] / total[0]).item(),
#                     "Net2 Top-5 Acc": (top5_num[1] / total[1]).item(),
#                     "Ensemble Top-5 Acc": (top5_num[2] / total[2]).item(),
#                     "eta": pbar.format_dict["elapsed"] * (len(pbar) / (pbar.n + 1)) - pbar.format_dict["elapsed"]
#                 })
#                 pbar.update(1)
#
#         acc1 = [round((top1_num[i] / total[i]).item(), 4) for i in range(args.number_net + 1)]
#         acc5 = [round((top5_num[i] / total[i]).item(), 4) for i in range(args.number_net + 1)]
#
#         with open(log_txt, 'a+') as f:
#             f.write('Test epoch:{}\t Test_loss_cls:{:.5f}\t Test top-1 accuracy:{}\t Test top-5 accuracy:{}\n'
#                     .format(epoch, test_loss_cls.avg, str(acc1), str(acc5)))
#
#         print('Test epoch:{}\t Test top-1 accuracy:{}\t Test top-5 accuracy:{}\n'.format(epoch, str(acc1), str(acc5)))
#
#     return max(acc1[:-1])
#
#
# if __name__ == '__main__':
#     best_acc = 0.  # best test accuracy
#     start_epoch = 0  # start from epoch 0 or last checkpoint epoch
#     criterion_ce = nn.CrossEntropyLoss()
#     criterion_decouple_ml = DecoupleMLLoss(args.alpha, args.beta, args.kd_T)
#
#     if args.evaluate:
#         print('Evaluate pre-trained weights from: {}'.format(args.evaluate_checkpoint))
#         checkpoint = torch.load(args.evaluate_checkpoint, map_location=torch.device('cpu'))
#         net.module.load_state_dict(checkpoint['net'])
#         best_acc = checkpoint['acc']
#         start_epoch = checkpoint['epoch'] + 1
#         test(start_epoch, criterion_ce)
#     else:
#         optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
#
#         criterion_list = [criterion_ce, criterion_decouple_ml]
#
#         if args.resume:
#             print('Resume pre-trained weights from: {}'.format(args.resume_checkpoint))
#             checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
#             net.module.load_state_dict(checkpoint['net'])
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             best_acc = checkpoint['acc']
#             start_epoch = checkpoint['epoch'] + 1
#
#         for epoch in range(start_epoch, args.epochs):
#             train(epoch, criterion_list, optimizer)
#             acc = test(epoch, criterion_ce)
#
#             state = {
#                 'net': net.module.state_dict(),
#                 'acc': acc,
#                 'epoch': epoch,
#                 'optimizer': optimizer.state_dict()
#             }
#             torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))
#
#             is_best = False
#             if best_acc < acc:
#                 best_acc = acc
#                 is_best = True
#
#             if is_best:
#                 shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
#                                 os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))
#
#         print('Evaluate the best model:')
#         print('load pre-trained weights from: {}'.format(
#             os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
#         args.evaluate = True
#         checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
#                                 map_location=torch.device('cpu'))
#         net.module.load_state_dict(checkpoint['net'])
#         start_epoch = checkpoint['epoch']
#         top1_acc = test(start_epoch, criterion_ce)
#
#         with open(log_txt, 'a+') as f:
#             f.write('Test top-1 best_accuracy: {} \n'.format(top1_acc))
#         print('Test top-1 best_accuracy: {} \n'.format(top1_acc))
