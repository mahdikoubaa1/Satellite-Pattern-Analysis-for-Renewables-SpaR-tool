'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """Dice binary crossentropy loss."""
    def __init__(self, batch=True):
        super(DiceBCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_pred, y_true)
        return a + b


class DiceLoss(nn.Module):
    """Dice loss implementation."""
    def __init__(self, batch=True):
        super(DiceLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.001
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            print(y_pred.shape)
            print(y_true.shape)
            intersection = torch.sum(y_pred * y_true)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_pred, y_true):
        loss = 1 - self.soft_dice_coeff(y_pred, y_true)
        return loss

    def __call__(self, y_pred, y_true):
        return self.soft_dice_loss(y_pred, y_true)
class focal_loss(nn.Module):
    def __init__(self,gamma=0.):
        super().__init__()
        self.gamma=gamma
    def forward(self,pred,target):
        ce=nn.functional.cross_entropy(pred,target,reduction='none')
        pred=nn.functional.softmax(pred,dim=1)
        t=(((1-pred)*target).sum(dim=1))**self.gamma
        x=ce*t
        x=x.mean()
        return x
class triplet_loss(nn.Module):
    def __init__(self,alpha=500):
        super().__init__()
        self.alpha=alpha
    def forward(self,pred,target):
        p=torch.moveaxis(pred,1,2)
        p=torch.reshape(p,[target.shape[2],-1])

        sp=[i for i in range(target.shape[2])if target[0,1,i,0]==1]
        nsp=[i for i in range(target.shape[2])if target[0,0,i,0]==1]
        dmatrix= (((p[:,None,:]-p)**2).sum(dim=2)+1e-40).sqrt()

        result=torch.cat(((dmatrix[sp][:,sp].max(dim=1)[0]-dmatrix[sp][:,nsp].min(dim=1)[0]+self.alpha).relu(),(dmatrix[nsp][:,nsp].max(dim=1)[0]-dmatrix[nsp][:,sp].min(dim=1)[0]+self.alpha).relu())).mean()
        
        return result

class BCELoss(nn.Module):
    """Binary crossentropy loss."""

    def __init__(self, batch=True):
        super(BCELoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.CrossEntropyLoss()

    def __call__(self, y_pred, y_true):
        loss = self.bce_loss(y_pred, y_true)
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self, batch=True):
        super(WeightedBCELoss, self).__init__()
        self.batch = batch
        weights = [0.3, 0.7]
        class_weights = torch.FloatTensor(weights).cuda()
        self.bce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def __call__(self, y_pred, y_true):
        loss = self.bce_loss(y_pred, y_true)
        return loss