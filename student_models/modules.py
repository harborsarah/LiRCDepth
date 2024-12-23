import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvModule, self).__init__()

        self.conv = torch.nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),\
                                        nn.ReLU())
    def forward(self, x):
        return self.conv(x)

class ConfModule(nn.Module):
    def __init__(self, params):
        super(ConfModule, self).__init__()

        out_channels = 16
        self.sparse = SparseModule(params, out_channels)

        self.denseaspp = _DenseASPPBlock(3, 16, out_channels)

        self.confidence = torch.nn.Sequential(nn.Conv2d(out_channels, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())


    def forward(self, image, radar):
        radar_feature = self.sparse(radar)
        image_feature = self.denseaspp(image)

        feature = radar_feature + image_feature
        confidence = self.confidence(feature)

        return confidence

class ConfModuleRad(nn.Module):
    def __init__(self, params):
        super(ConfModuleRad, self).__init__()

        out_channels = 16
        self.sparse = SparseModule(params, out_channels)

        self.confidence = torch.nn.Sequential(nn.Conv2d(out_channels, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())


    def forward(self, radar):
        radar_feature = self.sparse(radar)
        # image_feature = self.denseaspp(image)

        # feature = radar_feature + image_feature
        confidence = self.confidence(radar_feature)

        return confidence

class SparseModule(nn.Module):
    def __init__(self, params, out_channels):
        super(SparseModule, self).__init__()
        self.sparse_conv1 = SparseConv(params.radar_input_channels, 16, 7)
        self.sparse_conv2 = SparseConv(16, 16, 5)
        self.sparse_conv3 = SparseConv(16, 16, 3)
        self.sparse_conv4 = SparseConv(16, out_channels, 3)
        
    def forward(self, x):
        mask = (x[:, 0] > 0).float().unsqueeze(1)
        feature = x
        feature, mask = self.sparse_conv1(feature, mask)
        feature, mask = self.sparse_conv2(feature, mask)
        feature, mask = self.sparse_conv3(feature, mask)
        feature, mask = self.sparse_conv4(feature, mask)
        
        return feature

class _DenseASPPConv(nn.Sequential):
    def __init__(self, in_channels, inter_channels, out_channels, atrous_rate,
                 drop_rate=0.1, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPConv, self).__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, inter_channels, 1)),
        self.add_module('bn1', norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu1', nn.ReLU(True)),
        self.add_module('conv2', nn.Conv2d(inter_channels, out_channels, 3, dilation=atrous_rate, padding=atrous_rate)),
        self.add_module('bn2', norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs))),
        self.add_module('relu2', nn.ReLU(True)),
        self.drop_rate = drop_rate

    def forward(self, x):
        features = super(_DenseASPPConv, self).forward(x)
        if self.drop_rate > 0:
            features = F.dropout(features, p=self.drop_rate, training=self.training)
        return features


class _DenseASPPBlock(nn.Module):
    def __init__(self, in_channels, inter_channels1, inter_channels2,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_DenseASPPBlock, self).__init__()
        self.aspp_3 = _DenseASPPConv(in_channels, inter_channels1, inter_channels2, 3, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_6 = _DenseASPPConv(in_channels + inter_channels2 * 1, inter_channels1, inter_channels2, 6, 0.1,
                                     norm_layer, norm_kwargs)
        self.aspp_12 = _DenseASPPConv(in_channels + inter_channels2 * 2, inter_channels1, inter_channels2, 12, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_18 = _DenseASPPConv(in_channels + inter_channels2 * 3, inter_channels1, inter_channels2, 18, 0.1,
                                      norm_layer, norm_kwargs)
        self.aspp_24 = _DenseASPPConv(in_channels + inter_channels2 * 4, inter_channels1, inter_channels2, 24, 0.1,
                                      norm_layer, norm_kwargs)
        
        self.outconv = nn.Sequential(
            nn.Conv2d(in_channels+5*inter_channels2, inter_channels2, 1, 1, bias=False),
            nn.ReLU()
        )
    def forward(self, x):
        aspp3 = self.aspp_3(x)
        x = torch.cat([aspp3, x], dim=1)

        aspp6 = self.aspp_6(x)
        x = torch.cat([aspp6, x], dim=1)

        aspp12 = self.aspp_12(x)
        x = torch.cat([aspp12, x], dim=1)

        aspp18 = self.aspp_18(x)
        x = torch.cat([aspp18, x], dim=1)

        aspp24 = self.aspp_24(x)
        x = torch.cat([aspp24, x], dim=1)
        
        x = self.outconv(x)
        return x

class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 activation='relu'):
        super().__init__()

        padding = kernel_size//2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation == 'elu':
            self.act = nn.ELU()

        self.max_pool = nn.MaxPool2d(
            kernel_size, 
            stride=1, 
            padding=padding)

        

    def forward(self, x, mask):
        x = x*mask
        x = self.conv(x)
        normalizer = 1/(self.sparsity(mask)+1e-8)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.act(x)
        
        mask = self.max_pool(mask)

        return x, mask
