import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['IBDLoss', 'BCEDiceLoss', 'BCELoss', 'DiceLoss', 'LovaszHingeLoss', 'FocalLoss', 'DeepSupervisionLoss']

# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
#     wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wbce + wiou).mean()
#
#
# # PyTorch
# class IoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(IoULoss, self).__init__()
#
#     def forward(self, inputs, targets, smooth=1):
#         # comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)
#
#         # flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#
#         # intersection is equivalent to True Positive count
#         # union is the mutually inclusive area of all labels & predictions
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection
#
#         IoU = (intersection + smooth) / (union + smooth)
#
#         return 1 - IoU

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

class IBDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)

        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        union = (input + target)
        iou = (intersection.sum(1) + 1) / (union.sum(1) - intersection.sum(1) + 1)
        iou = 1 - iou.sum() / num
        return 0.5 * bce + dice + 0.5 * iou
# 普通对比
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        bce = F.binary_cross_entropy_with_logits(input, target)

        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice




# 自己的方法
# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input, target):
#         # 假设 target 是 [8, 1, 384, 384]
#         target = target.squeeze(1)  # 将 [8, 1, 384, 384] 转为 [8, 384, 384]
#
#         # 转为 one-hot 编码形式，变成 [8, 2]
#         target_one_hot = F.one_hot(target.long(), num_classes=2)  # 变成 [8, 2, 384, 384]
#         target_one_hot = target_one_hot.view(target_one_hot.size(0), -1)  # 转为 [8, 2 * 384 * 384]
#
#         # 对于二分类，取每个样本的每一类的平均
#         target_one_hot = target_one_hot.sum(dim=1).view(target.size(0), -1)  # 将其变为 [8, 2]
#
#         bce = F.binary_cross_entropy_with_logits(input, target_one_hot)
#
#         smooth = 1e-5
#         input = torch.sigmoid(input)
#         num = target.size(0)
#         input = input.view(num, -1)
#         intersection = (input * target_one_hot).sum(1)
#         dice = (2. * intersection + smooth) / (input.sum(1) + target_one_hot.sum(1) + smooth)
#         dice = 1 - dice.mean()  # 取平均
#         return 0.5 * bce + dice

class BCELoss1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        return bce

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight=weight, size_average=size_average)

    def forward(self, pred, target):
        size = pred.size(0)
        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        loss = self.bceloss(pred_flat, target_flat)

        return loss


"""Dice loss"""


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1

        size = pred.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # pred = F.sigmoid(pred)

        # flatten label and prediction tensors
        pred = pred.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (pred * targets).sum()
        total = (pred + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


"""BCE + DICE Loss"""


class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = diceloss + bceloss

        return loss


"""BCE + IoU Loss"""


class BceIoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceIoULoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.iou = IoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        iouloss = self.iou(pred, target)

        loss = iouloss + bceloss

        return loss


""" Structure Loss: https://github.com/DengPingFan/PraNet/blob/master/MyTrain.py """


class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


""" Deep Supervision Loss"""


class DeepSupervisionLoss(nn.Module):
    def __init__(self, typeloss="BceDiceLoss"):
        super(DeepSupervisionLoss, self).__init__()

        if typeloss == "BceDiceLoss":
            self.criterion = BceDiceLoss()
        elif typeloss == "BceIoULoss":
            self.criterion = BceIoULoss()
        elif typeloss == "StructureLoss":
            self.criterion = StructureLoss()
        else:
            raise Exception("Loss name is unvalid.")

    def forward(self, pred, gt):
        d0, d1, d2, d3, d4 = pred[0:]
        loss0 = self.criterion(torch.sigmoid(d0), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss1 = self.criterion(torch.sigmoid(d1), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss2 = self.criterion(torch.sigmoid(d2), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss3 = self.criterion(torch.sigmoid(d3), gt)
        gt = F.interpolate(gt, scale_factor=0.5, mode='bilinear', align_corners=True)
        loss4 = self.criterion(torch.sigmoid(d4), gt)

        return loss0 + loss1 + loss2 + loss3 + loss4
