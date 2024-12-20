import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3_3D(nn.Module):
    def __init__(self, depth, in_channels, out_classes, resnet='resnet34_os8', last_activation = None):
        super(DeepLabV3_3D, self).__init__()
        self.out_classes = out_classes
        self.last_activation = last_activation

        if resnet.lower() == 'resnet18_os16':
            self.resnet = ResNet18_OS16(in_channels)
        
        elif resnet.lower() == 'resnet34_os16':
            self.resnet = ResNet34_OS16(in_channels)
        
        elif resnet.lower() == 'resnet50_os16':
            self.resnet = ResNet50_OS16(in_channels)
        
        elif resnet.lower() == 'resnet101_os16':
            self.resnet = ResNet101_OS16(in_channels)
        
        elif resnet.lower() == 'resnet152_os16':
            self.resnet = ResNet152_OS16(in_channels)
        
        elif resnet.lower() == 'resnet18_os8':
            self.resnet = ResNet18_OS8(in_channels)
        
        elif resnet.lower() == 'resnet34_os8':
            self.resnet = ResNet34_OS8(in_channels)

        if resnet.lower() in ['resnet50_os16', 'resnet101_os16', 'resnet152_os16']:
            self.aspp = ASPP_Bottleneck(num_classes=self.out_classes)
        else:
            self.aspp = ASPP(num_classes=self.out_classes)

        self.final_conv = torch.nn.Conv3d(depth, 1, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):

        h = x.size()[2]
        w = x.size()[3]
        c = x.size()[4]

        feature_map = self.resnet(x)

        output = self.aspp(feature_map)

        output = F.interpolate(output, size=(h, w, c), mode='trilinear', align_corners=True)
        
        if self.last_activation != None:
            if self.last_activation.lower() == 'sigmoid':
                output = nn.Sigmoid()(output)
            elif self.last_activation.lower() == 'softmax':
                output = nn.Softmax()(output)
        
        # output = torch.permute(output, (0,2,1,3,4))
        output = output.permute((0,2,1,3,4))
        output = self.final_conv(output)
        output = torch.squeeze(output)
        
        
        return output

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input_channels, block, layers, num_classes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(input_channels, **kwargs):
    model = ResNet(input_channels, BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(input_channels, **kwargs):
    model = ResNet(input_channels, Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


# -------------------------------------- Resnet for Deeplab --------------------------------------
def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1)

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels
        
        if type(dilation) != type(1):
            dilation = 1
            
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm3d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out + self.downsample(x)

        out = F.relu(out)

        return out

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_Bottleneck_OS16, self).__init__()

        if num_layers == 50:
            resnet = resnet50(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 101:
            resnet = resnet101(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        elif num_layers == 152:
            resnet = resnet152(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck, in_channels=4*256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output

class ResNet_BasicBlock_OS16(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_BasicBlock_OS16, self).__init__()

        if num_layers == 18:
            resnet = resnet18(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 2

        elif num_layers == 34:
            resnet = resnet34(input_channels)
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")
    
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        c4 = self.resnet(x)

        output = self.layer5(c4)

        return output

class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers, input_channels):
        super(ResNet_BasicBlock_OS8, self).__init__()

        if num_layers == 18:
            resnet = resnet18(input_channels)
            
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2

        elif num_layers == 34:
            resnet = resnet34(input_channels)
            
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)

        

    def forward(self, x):
        c3 = self.resnet(x)

        output = self.layer4(c3)
        output = self.layer5(output)
        

        return output

def ResNet18_OS16(input_channels):
    return ResNet_BasicBlock_OS16(num_layers=18, input_channels=input_channels)

def ResNet50_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=50, input_channels=input_channels)

def ResNet101_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=101, input_channels=input_channels)

def ResNet152_OS16(input_channels):
    return ResNet_Bottleneck_OS16(num_layers=152, input_channels=input_channels)

def ResNet34_OS16(input_channels):
    return ResNet_BasicBlock_OS16(num_layers=34, input_channels=input_channels)

def ResNet18_OS8(input_channels):
    return ResNet_BasicBlock_OS8(num_layers=18, input_channels=input_channels)

def ResNet34_OS8(input_channels):
    return ResNet_BasicBlock_OS8(num_layers=34, input_channels=input_channels)


class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv3d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(256)

        self.conv_3x3_1 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(256)

        self.conv_3x3_2 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(256)

        self.conv_3x3_3 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm3d(256)

        self.conv_1x1_3 = nn.Conv3d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm3d(256)

        self.conv_1x1_4 = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        feature_map_c = feature_map.size()[4]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w, feature_map_c), mode='trilinear', align_corners=True)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)


        return out

class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes):
        super(ASPP_Bottleneck, self).__init__()

        self.conv_1x1_1 = nn.Conv3d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm3d(256)

        self.conv_3x3_1 = nn.Conv3d(4*512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm3d(256)

        self.conv_3x3_2 = nn.Conv3d(4*512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm3d(256)

        self.conv_3x3_3 = nn.Conv3d(4*512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm3d(256)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv_1x1_2 = nn.Conv3d(4*512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm3d(256)

        self.conv_1x1_3 = nn.Conv3d(1280, 256, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm3d(256)

        self.conv_1x1_4 = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):
        feature_map_h = feature_map.size()[2]
        feature_map_w = feature_map.size()[3]
        feature_map_c = feature_map.size()[4]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))

        out_img = self.avg_pool(feature_map)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w, feature_map_c), mode='trilinear', align_corners=True)
        
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))
        out = self.conv_1x1_4(out)

        return out

