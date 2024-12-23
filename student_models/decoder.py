import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import torchvision.models as models
from student_models.modules import ConvModule

class InvertedResidualBlockTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=1):
        super(InvertedResidualBlockTranspose, self).__init__()

        self.stride = stride

        self.expansion_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True)
        )

        self.depthwise_layer_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels * t, in_channels * t, kernel_size=3, stride=stride, padding=1,
                               output_padding=(0 if stride == 1 else 1), groups=in_channels * t, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True)
        )

        self.pointwise_layer_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels * t, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.expansion_layer(x)
        out = self.depthwise_layer_transpose(out)
        out = self.pointwise_layer_transpose(out)

        return out
    

class decoder_student(nn.Module):
    def __init__(self, params, feat_out_channels, feat_out_channels_radar, num_features=[512, 128, 128, 64, 32]):
        super(decoder_student, self).__init__()
        self.params = params
        self.feat_out_channels = feat_out_channels
        self.feat_out_channels_radar = feat_out_channels_radar

        self.num_features = num_features

        # if both w_radar_feature_distill and w_image_feature_distill are > 0, we need to make the channels same, to perform elementwise addtion fusion
        if params.w_image_feature_distill > 0 and params.w_radar_feature_distill > 0:
            self.conv_radar = nn.ModuleList([
                ConvModule(feat_out_channels_radar[-5], feat_out_channels[-5], 1, 1),
                ConvModule(feat_out_channels_radar[-4], feat_out_channels[-4], 1, 1),
                ConvModule(feat_out_channels_radar[-3], feat_out_channels[-3], 1, 1),
                ConvModule(feat_out_channels_radar[-2], feat_out_channels[-2], 1, 1),
                ConvModule(feat_out_channels_radar[-1], feat_out_channels[-1], 1, 1),
            ])


        if self.params.radar_confidence:
            self.pool5 = torch.nn.AvgPool2d(32, 32)
            self.pool4 = torch.nn.AvgPool2d(16, 16)
            self.pool3 = torch.nn.AvgPool2d(8, 8)
            self.pool2 = torch.nn.AvgPool2d(4, 4)
            self.pool1 = torch.nn.AvgPool2d(2, 2)

        # H/32 -> H/16
        i = 0
        self.upconv5 = nn.Sequential(
            InvertedResidualBlockTranspose(feat_out_channels[-1], num_features[i]//16, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//16, num_features[i]//16, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//16, num_features[i], 6, 2))
        
        i += 1

        self.upconv4 = nn.Sequential(
            InvertedResidualBlockTranspose(num_features[i-1]+feat_out_channels[-2], num_features[i]//8, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//8, num_features[i]//8, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//8, num_features[i]//8, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//8, num_features[i]//8, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//8, num_features[i]//8, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//8, num_features[i]//8, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//8, num_features[i]//8, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//8, num_features[i], 6, 2))
        self.scale8 = nn.Sequential(
            nn.Conv2d(num_features[i], 1, 1, 1, bias=False),
            nn.ReLU()
        )
        
        i += 1
        self.upconv3 = nn.Sequential(
            InvertedResidualBlockTranspose(num_features[i-1]+feat_out_channels[-3], num_features[i]//4, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//4, num_features[i]//4, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//4, num_features[i], 6, 2)
        )
        self.scale4 = nn.Sequential(
            nn.Conv2d(num_features[i], 1, 1, 1, bias=False),
            nn.ReLU()
        )

        i += 1
        self.upconv2 = nn.Sequential(
            InvertedResidualBlockTranspose(num_features[i-1]+feat_out_channels[-4], num_features[i]//2, 6, 1),
            InvertedResidualBlockTranspose(num_features[i]//2, num_features[i], 6, 2)
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(num_features[i], 1, 1, 1, bias=False),
            nn.ReLU()
        )

        i += 1
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(num_features[i-1]+feat_out_channels[-5], num_features[i], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        )
        self.scale1 = nn.Sequential(
            nn.Conv2d(num_features[i], 1, 1, 1, bias=False),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_features[i]+4, num_features[i], 3, 1, 1, bias=False),
            nn.ReLU()
        )
        self.get_depth = torch.nn.Sequential(nn.Conv2d(num_features[i], 1, 3, 1, 1, bias=False),
                                               nn.Sigmoid())

    def forward(self, img_features, rad_features, radar_confidence=None):
        feat = []
        depth = []
        if self.params.w_image_feature_distill > 0 and self.params.w_radar_feature_distill > 0:
            rad_features_new = []
            for i in range(len(rad_features)):
                rad_features_new.append(self.conv_radar[i](rad_features[i]))
            rad_features = rad_features_new
        # 1/32 -> 1/16
        if radar_confidence is not None:
            radar_confidence5 = self.pool5(radar_confidence)
            radar_confidence4 = self.pool4(radar_confidence)
            radar_confidence3 = self.pool3(radar_confidence)
            radar_confidence2 = self.pool2(radar_confidence)
            radar_confidence1 = self.pool1(radar_confidence)

        if radar_confidence is not None:
            feat5 = img_features[-1] + rad_features[-1]*radar_confidence5
        else:
            feat5 = img_features[-1] + rad_features[-1]
        up_feat5 = self.upconv5(feat5)
        feat.append(up_feat5)

        # 1/16 -> 1/8
        if radar_confidence is not None:
            up_feat5_ = torch.cat((up_feat5, img_features[-2]+rad_features[-2]*radar_confidence4), dim=1)
        else:
            up_feat5_ = torch.cat((up_feat5, img_features[-2]+rad_features[-2]), dim=1)
        up_feat4 = self.upconv4(up_feat5_)
        depth_8x8_scaled = self.scale8(up_feat4) # 1/8
        depth_8x8_scaled_us = torch_nn_func.interpolate(depth_8x8_scaled, scale_factor=8, mode='nearest') # 1
        feat.append(up_feat4)
        depth.append(depth_8x8_scaled_us)

        # 1/8 -> 1/4
        if radar_confidence is not None:
            up_feat4_ = torch.cat((up_feat4, img_features[-3]+rad_features[-3]*radar_confidence3), dim=1)
        else:
            up_feat4_ = torch.cat((up_feat4, img_features[-3]+rad_features[-3]), dim=1)
        up_feat3 = self.upconv3(up_feat4_)
        depth_4x4_scaled = self.scale4(up_feat3) # 1/4
        depth_4x4_scaled_us = torch_nn_func.interpolate(depth_4x4_scaled, scale_factor=4, mode='nearest') # 1
        feat.append(up_feat3)
        depth.append(depth_4x4_scaled_us)

        # 1/4 -> 1/2
        if radar_confidence is not None:
            up_feat3_ = torch.cat((up_feat3, img_features[-4]+rad_features[-4]*radar_confidence2), dim=1)
        else:
            up_feat3_ = torch.cat((up_feat3, img_features[-4]+rad_features[-4]), dim=1)
        up_feat2 = self.upconv2(up_feat3_)
        depth_2x2_scaled = self.scale2(up_feat2)
        depth_2x2_scaled_us = torch_nn_func.interpolate(depth_2x2_scaled, scale_factor=2, mode='nearest') # 1
        feat.append(up_feat2)
        depth.append(depth_2x2_scaled_us)

        # 1/2 ->1
        if radar_confidence is not None:
            up_feat2_ = torch.cat((up_feat2, img_features[-5]+rad_features[-5]*radar_confidence1), dim=1)
        else:
            up_feat2_ = torch.cat((up_feat2, img_features[-5]+rad_features[-5]), dim=1)
        up_feat1 = self.upconv1(up_feat2_)
        depth_1x1_scaled = self.scale1(up_feat1)
        up_feat1 = self.conv1(torch.cat([up_feat1, depth_8x8_scaled_us, depth_4x4_scaled_us, depth_2x2_scaled_us, depth_1x1_scaled], dim=1))
        feat.append(up_feat1)
        # depth.append(depth_1x1_scaled)

        final_depth = self.params.max_depth * self.get_depth(up_feat1)

        return feat, depth, final_depth