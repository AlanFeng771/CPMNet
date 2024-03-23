import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, groups=1,
                 norm_type='none', act_type='ReLU'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                              padding=kernel_size // 2 + dilation - 1, dilation=dilation, bias=False)
        self.norm = norm_layer3d(norm_type, out_channels)
        self.act = act_layer(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

def act_layer(act='ReLU'):
    if act == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act == 'LeakyReLU':
        return nn.LeakyReLU(inplace=True, negative_slope=0.1)
    elif act == 'ELU':
        return nn.ELU(inplace=True)
    elif act == 'PReLU':
        return nn.PReLU(inplace=True)
    elif act == 'RRelu':
        return nn.RReLU(inplace=True)
    else:
        return Identity()

def norm_layer3d(norm_type, num_features):
    if norm_type == 'batchnorm':
        return nn.BatchNorm3d(num_features=num_features, momentum=0.05)
    elif norm_type == 'instancenorm':
        return nn.InstanceNorm3d(num_features=num_features, affine=True)
    elif norm_type == 'groupnorm':
        return nn.GroupNorm(num_groups=num_features // 8, num_channels=num_features)
    else:
        return Identity()

class SCConv3D(nn.Module):
    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super(SCConv3D, self).__init__()
        self.k2 = nn.Sequential(
                    nn.AvgPool3d(kernel_size=pooling_r, stride=pooling_r), 
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x
        k2 = F.interpolate(self.k2(x), identity.size()[2:])
        out = torch.sigmoid(torch.add(identity, k2)) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

# class SCBottleneck(nn.Module):
#     """SCNet SCBottleneck
#     """
#     expansion = 4
#     pooling_r = 4 # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

#     def __init__(self, inplanes, planes, stride=1, downsample=None,
#                  cardinality=1, bottleneck_width=32,
#                  avd=False, dilation=1, is_first=False,
#                  norm_layer=None, norm_type=None, act_type=None):
#         super(SCBottleneck, self).__init__()
#         group_width = int(planes * (bottleneck_width / 64.)) * cardinality
#         self.conv1_a = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
# #         self.bn1_a = norm_layer(group_width)
#         self.norm1_a = norm_layer3d(norm_type, group_width)
#         self.conv1_b = nn.Conv3d(inplanes, group_width, kernel_size=1, bias=False)
# #         self.bn1_b = norm_layer(group_width)
#         self.norm1_b = norm_layer3d(norm_type, group_width)
#         self.avd = avd and (stride > 1 or is_first)
# #         self.norm = norm_layer3d(norm_type, out_channels)
#         self.act = act_layer(act_type)
#         if self.avd:
#             self.avd_layer = nn.AvgPool3d(3, stride, padding=1)
#             stride = 1

#         self.k1 = nn.Sequential(
#                     nn.Conv3d(
#                         group_width, group_width, kernel_size=3, stride=stride,
#                         padding=dilation, dilation=dilation,
#                         groups=cardinality, bias=False),
#                     norm_layer(group_width),
#                     )

#         self.scconv = SCConv3D(
#             group_width, group_width, stride=stride,
#             padding=dilation, dilation=dilation,
#             groups=cardinality, pooling_r=self.pooling_r, norm_layer=norm_layer)

#         self.conv3 = nn.Conv3d(
#             group_width * 2, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = norm_layer(planes*4)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.dilation = dilation
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out_a= self.conv1_a(x)
#         out_a = self.norm1_a(out_a)
#         out_b = self.conv1_b(x)
#         out_b = self.norm1_b(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)

#         out_a = self.k1(out_a)
#         out_b = self.scconv(out_b)
#         out_a = self.relu(out_a)
#         out_b = self.relu(out_b)

#         if self.avd:
#             out_a = self.avd_layer(out_a)
#             out_b = self.avd_layer(out_b)

#         out = self.conv3(torch.cat([out_a, out_b], dim=1))
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out