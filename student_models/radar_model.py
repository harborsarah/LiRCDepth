import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import torchvision.models as models
from student_models.modules import ConvModule

class encoder_radar_student(nn.Module):
    def __init__(self, params):
        super(encoder_radar_student, self).__init__()
        self.params = params
        self.conv = nn.Sequential(
            nn.Conv2d(params.radar_input_channels, 3, 3, 1, 1, bias=False),
            nn.ReLU()
        )
        self.base_model = models.mobilenet_v2(pretrained=True).features
        self.feat_inds = [2, 4, 7, 11, 19]
        self.feat_out_channels = [16, 24, 32, 64, 1280]
        self.feat_names = []

        if params.w_radar_feature_distill > 0 and params.w_image_feature_distill > 0:
            self.feat_out_channels_radar = [64, 128, 128, 512, 512]
            self.conv_radar = nn.ModuleList(
                [
                    ConvModule(self.feat_out_channels[0], self.feat_out_channels_radar[0], 1, 1),
                    ConvModule(self.feat_out_channels[1], self.feat_out_channels_radar[1], 1, 1),
                    ConvModule(self.feat_out_channels[2], self.feat_out_channels_radar[2], 1, 1),
                    ConvModule(self.feat_out_channels[3], self.feat_out_channels_radar[3], 1, 1),
                    ConvModule(self.feat_out_channels[4], self.feat_out_channels_radar[4], 1, 1),
                    
                ])
            self.feat_out_channels = self.feat_out_channels_radar
        elif params.w_radar_feature_distill:
            self.feat_out_channels_radar = [64, 128, 128, 512, 512]
            self.conv_radar = nn.ModuleList(
                [
                    ConvModule(self.feat_out_channels[0], self.feat_out_channels_radar[0], 1, 1),
                    ConvModule(self.feat_out_channels[1], self.feat_out_channels_radar[1], 1, 1),
                    ConvModule(self.feat_out_channels[2], self.feat_out_channels_radar[2], 1, 1),
                    ConvModule(self.feat_out_channels[3], self.feat_out_channels_radar[3], 1, 1),
                    ConvModule(self.feat_out_channels[4], self.feat_out_channels_radar[4], 1, 1),
                    
                ])
            self.feat_out_channels = self.feat_out_channels_radar
        elif params.w_image_feature_distill:
            self.feat_out_channels_radar = [64, 64, 128, 256, 512]
            self.conv_radar = nn.ModuleList(
                [
                    ConvModule(self.feat_out_channels[0], self.feat_out_channels_radar[0], 1, 1),
                    ConvModule(self.feat_out_channels[1], self.feat_out_channels_radar[1], 1, 1),
                    ConvModule(self.feat_out_channels[2], self.feat_out_channels_radar[2], 1, 1),
                    ConvModule(self.feat_out_channels[3], self.feat_out_channels_radar[3], 1, 1),
                    ConvModule(self.feat_out_channels[4], self.feat_out_channels_radar[4], 1, 1),
                    
                ])
            self.feat_out_channels = self.feat_out_channels_radar



    def forward(self, x):
        feature = x
        feature = self.conv(feature)
        skip_feat = []
        i = 1
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(feature)
            if i == 2 or i == 4 or i == 7 or i == 11 or i == 19:
                skip_feat.append(feature)

            i = i + 1
        
        if self.params.w_radar_feature_distill or self.params.w_image_feature_distill:
            new_skip_feat = []
            for i, feat in enumerate(skip_feat):
                new_skip_feat.append(self.conv_radar[i](feat))
            return new_skip_feat

        return skip_feat