import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, rm, psm, pos_equal_one, neg_equal_one, targets):
        # print('look shape111111111111111111111111111111s:{}   {}   {}'.format(pos_equal_one.shape,neg_equal_one.shape,targets.shape))
        # print('look shape222222222222222222222222222222s:{}   {}'.format(rm.shape,psm.shape))
        # p_pos = F.sigmoid(psm.view(4,rm.size(1),int(rm.size(2)/4),rm.size(3)))
        # psm = psm.view(4,rm.size(1), -1,rm.size(3))
        p_pos = torch.sigmoid(psm)
        rm = rm.contiguous()
        # print(targets.shape)
        # rm = rm.view(4,rm.size(1),int(rm.size(2)/4),-1,7)
        rm = rm.view(rm.size(0),rm.size(1), rm.size(2),-1,7)
        # rm = rm.view(4,rm.size(1), -1,rm.size(3),rm .size(4))
        targets = targets.view(targets.size(0),targets.size(1),targets.size(2),-1,7)
        pos_equal_one_for_reg = pos_equal_one.unsqueeze(pos_equal_one.dim()).expand(-1,-1,-1,-1,7)
        rm_pos = rm * pos_equal_one_for_reg
        targets_pos = targets * pos_equal_one_for_reg

        cls_pos_loss = -pos_equal_one * torch.log(p_pos + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (pos_equal_one.sum() + 1e-6)

        cls_neg_loss = -neg_equal_one * torch.log(1 - p_pos + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (neg_equal_one.sum() + 1e-6)

        reg_loss = self.smoothl1loss(rm_pos, targets_pos)
        reg_loss = reg_loss / (pos_equal_one.sum() + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss

        return conf_loss, reg_loss












