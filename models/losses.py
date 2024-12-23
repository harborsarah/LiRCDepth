import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import concurrent.futures


class similarity_loss(nn.Module):
    def __init__(self, scale=0):
        super(similarity_loss, self).__init__()
        self.scale = scale
    
    def L2(self, f_):
        return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8
    
    def similarity(self, feat):
        feat = feat.float()
        tmp = self.L2(feat).detach()
        feat = feat/tmp
        feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
        return torch.einsum('icm,icn->imn', [feat, feat])

    def sim_dis_compute(self, f_S, f_T):
        sim_err = ((self.similarity(f_T) - self.similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
        sim_dis = sim_err.sum()
        return sim_dis
    
    def forward(self, feat_S, feat_T):
        if self.scale != 0:
            total_w, total_h = feat_T.shape[2], feat_T.shape[3]
            patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
            maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
            feat_S = maxpool(feat_S)
            feat_T = maxpool(feat_T)
        
        loss = self.sim_dis_compute(feat_S, feat_T)
        return loss



class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask, uncertainty=None):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

class l1_loss(nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask=None, uncertainty=None):
        loss = torch.nn.functional.l1_loss(depth_est, depth_gt, reduction='none')
        if uncertainty is not None and mask is not None:
            mask = mask * uncertainty
            loss = mask * loss
            loss = torch.sum(loss) / torch.sum(mask+1e-7)
        elif mask is not None:
            loss = mask * loss
            loss = torch.sum(loss) / torch.sum(mask)
        elif uncertainty is not None:
            h = uncertainty.shape[2]
            h_feat = depth_est.shape[2]
            if h_feat != h:
                scale = int(h/h_feat)
                uncertainty = torch.nn.functional.avg_pool2d(uncertainty, scale, scale)
            loss = uncertainty * loss
            # loss = torch.sum(loss) / torch.sum(uncertainty+1e-7)
            loss = torch.mean(loss)

        else:
            loss = torch.nn.functional.l1_loss(depth_est, depth_gt)
        return loss

class l2_loss(nn.Module):
    def __init__(self):
        super(l2_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.mse_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class binary_cross_entropy(nn.Module):
    def __init__(self):
        super(binary_cross_entropy, self).__init__()

    def forward(self, confidence, radar_gt, mask):
        loss = torch.nn.functional.binary_cross_entropy(confidence, radar_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class smoothness_loss_func(nn.Module):
    def __init__(self):
        super(smoothness_loss_func, self).__init__()
    
    def gradient_yx(self, T):
        '''
        Computes gradients in the y and x directions

        Arg(s):
            T : tensor
                N x C x H x W tensor
        Returns:
            tensor : gradients in y direction
            tensor : gradients in x direction
        '''

        dx = T[:, :, :, :-1] - T[:, :, :, 1:]
        dy = T[:, :, :-1, :] - T[:, :, 1:, :]
        return dy, dx
    
    def forward(self, predict, image):
        predict_dy, predict_dx = self.gradient_yx(predict)
        image_dy, image_dx = self.gradient_yx(image)

        # Create edge awareness weights
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))
        
        return smoothness_x + smoothness_y

