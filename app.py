import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
import os

from collections import Counter

from pathlib import Path
import random

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision
import torch.optim as optim
from functools import partial

import torch.nn as nn

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias,stride=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
from torch.autograd import Variable
class Learned_Dw_Conv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, fiter_kernel=3, stride=1, padding=1, dropout_rate=0.3, k=2,cardinality=32):
        super(Learned_Dw_Conv, self).__init__()

        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.group=self.out_channels//self.cardinality
        self.in_channel_per_group=in_channels//self.group
        self.dwconv = nn.Conv2d(in_channels, out_channels, fiter_kernel, stride, padding, bias=False,groups=self.group)

        self.delta_prune = (int)(self.in_channel_per_group*(self.cardinality-k)*0.25)

        self.tmp = self.group*(self.in_channel_per_group*self.cardinality- 4*self.delta_prune)
        self.pwconv = nn.Conv2d(self.tmp, out_channels, 1, 1, bias=False)

        self.dwconv2 = nn.Conv2d(self.tmp, self.tmp, fiter_kernel, stride, padding, groups=self.tmp, bias=False)
        self.register_buffer('index', torch.LongTensor(self.tmp))
        self.register_buffer('_mask_dw', torch.ones(self.dwconv.weight.size()))
        self.register_buffer('_count', torch.zeros(1))
        self.drop=nn.Dropout(0.3)
        #self.pwconv.weight.requires_grad = False
        #self.dwconv2.weight.requires_grad = False
    def _check_drop(self):
        progress = Learned_Dw_Conv.global_progress
        if progress == 0:
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress<45 :
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress>45 :
            self.dwconv.weight.data.zero_()
            ### Check for dropping
        if progress == 11 or progress == 23 or progress == 34 or progress == 45: # or progress==150 :
            # if progress == 1 or progress == 2 or progress == 3 or progress == 4:
            if progress<=45:
                self._dropping_group(self.delta_prune)
         #   else:
         #       if self.in_channel_per_group==8:
        #            self._dropping_group(16)
         #       else:
        #           self._dropping_group(32)
        return
    def _dropping_group(self,delta):
        if Learned_Dw_Conv.global_progress <= 45:
            weight=self.dwconv.weight*self.mask_dw
            weight=weight.view(self.group,self.cardinality,self.in_channel_per_group,3,3).abs().sum([3,4])
            for i in range(self.group):
                weight_tmp=weight[i,:,:].view(-1)
                di=weight_tmp.sort()[1][self.count:self.count+delta]
                for d in di.data:
                    out_ = d // self.in_channel_per_group
                    in_ = d % self.in_channel_per_group
                    self._mask_dw[i*self.cardinality+out_, in_, :, :].fill_(0)
            self.count = self.count + delta
            #print(self.in_channel_per_group)
            #print(self.delta_prune)
            #print(self.count)
        index=0
        if Learned_Dw_Conv.global_progress == 45:
            self.pwconv.weight.data.zero_()
            for i in range(self.group):
                for j in range(self.cardinality):
                    for k in range(self.in_channel_per_group):
                        if self._mask_dw[i*self.cardinality+j,k,0,0]==1:
                            self.index[index]=i*self.in_channel_per_group+k
                            self.dwconv2.weight.data[index,:,:,:]=self.dwconv.weight.data[i*self.cardinality+j,k,:,:].view(1,3,3)
                            self.pwconv.weight.data[i*self.cardinality+j,index,:,:].fill_(1)
                            index=index+1
            assert index==self.tmp
            self.dwconv.weight.data.zero_()
    def forward(self, x):
        progress = Learned_Dw_Conv.global_progress
        self._check_drop()
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output

        if progress < 45:
            weight = self.dwconv.weight * self.mask_dw
            return F.conv2d(x,weight, None, self.dwconv.stride,
                            1, self.dwconv.dilation, self.group)
        else:
            x = torch.index_select(x, 1, Variable(self.index))
            x = self.dwconv2(x)
            self.pwconv.weight.data = self.pwconv.weight.data  # *self.mask_pw
            x = F.conv2d(x, self.pwconv.weight, None, self.pwconv.stride,
                         0, self.pwconv.dilation, 1)
            return x

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def mask_dw(self):
        return Variable(self._mask_dw)

    @property
    def mask_pw(self):
        return Variable(self._mask_pw)
    @property
    def ldw_loss(self):
        if Learned_Dw_Conv.global_progress >= 45:
            return 0
        weight = self.dwconv.weight * self.mask_dw
        weight_1=weight.abs().sum(-1).sum(-1).view(self.group,self.cardinality,self.in_channel_per_group)
        weight=weight.abs().sum([2,3]).view(self.group,-1)
        mask=torch.ge(weight,torch.topk(weight,2*self.in_channel_per_group,1,sorted=True)[0][:,2*self.in_channel_per_group-1]
                      .view(self.group,1).expand_as(weight)).view(self.group,self.cardinality,self.in_channel_per_group)\
            .sum(1).view(self.group,1,self.in_channel_per_group)
        mask = torch.exp((mask.float() - 1.5 * self.k) / (10)) - 1
        mask=mask.expand_as(weight_1)
        weight=(weight_1.pow(2)*mask).sum(1).clamp(min=1e-6).sum(-1).sum(-1)
        return weight

import torchvision.models as models
from torch import nn

# Load the EfficientNet model (for example, EfficientNet-B0)
model1 = models.squeezenet1_0(pretrained=True)

model1=nn.Sequential(*list(model1.features.children())[:-1])


class ChannelShuffle2(nn.Module):
    def __init__(self, channels, groups):
        super(ChannelShuffle2, self).__init__()
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle2(x, self.groups)

def channel_shuffle2(x, groups):
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

""" Conv2d w/ Same Padding

Hacked together by Ross Wightman
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Callable
import math



def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
    
""" PyTorch Mixed Convolution

Paper: MixConv: Mixed Depthwise Convolutional Kernels (https://arxiv.org/abs/1907.09595)

Hacked together by Ross Wightman
"""

import torch
from torch import nn as nn




def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution

    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            conv_groups = out_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                create_conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=dilation, groups=conv_groups, **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CBAM']


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = self.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out)
        return x * scale.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model1=model1
        


        self.conv1=nn.Conv2d(512,513,kernel_size=3,padding=1,stride=1)
        self.ldsconv1=Learned_Dw_Conv(in_channels=513, out_channels=522, fiter_kernel=3, stride=1, padding=1, dropout_rate=0.3, k=2,cardinality=174)
        self.mix1=MixedConv2d(522,525)
        self.eca1=eca_layer(525)
        self.dwsc1=depthwise_separable_conv(525,528)
        self.cs1=ChannelShuffle2(525,3)
        self.groupconv1=nn.Conv2d(525,528,3,padding=1, groups=3)
        self.ghost1=GhostBottleneck(528,531,534,stride=2)
      
        
        self.ldsconv2=Learned_Dw_Conv(in_channels=534, out_channels=537, fiter_kernel=3, stride=1, padding=1, dropout_rate=0.3, k=2,cardinality=179)
        self.mix2=MixedConv2d(537,540)
        self.eca2=eca_layer(540)
        self.dwsc2=depthwise_separable_conv(540,543)
        self.cs2=ChannelShuffle2(543,3)
        self.groupconv2=nn.Conv2d(540,543,3,padding=1, groups=3)
        self.ghost2=GhostBottleneck(543,546,549,stride=1)


        self.eca0=CBAM(549)

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.activation = nn.ReLU()
        self.dropout=nn.Dropout(0.5)
        self.fc1 = nn.Linear(549,6)  

    

            
        
    def forward(self, x):
        x=self.model1(x)
        x=self.conv1(x)
        x=self.ldsconv1(x)
        x=self.mix1(x)
        x=self.eca1(x)
        a=self.dwsc1(x)
        b=self.cs1(x)
        b=self.groupconv1(x)
        c=torch.add(a,b)
        c=self.ghost1(c)

        x=self.ldsconv2(c)
        x=self.mix2(x)
        x=self.eca2(x)
        a=self.dwsc2(x)
        b=self.cs2(x)
        b=self.groupconv2(x)
        c=torch.add(a,b)
        c=self.ghost2(c)

        c=self.eca0(c)
        x= self.global_pool(c)
        x = torch.flatten(x, start_dim=1)
        x = self.activation(x)
        x=self.dropout(x)
        x = self.fc1(x)

        

        return x

import torch
from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms


# Initialize the Flask app
app = Flask(__name__)
model = CustomModel()
model.load_state_dict(torch.load('./best_model_weights.pth', map_location=torch.device('cpu')))
model.eval()  

# Define image transformation (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust to the size your model expects
    transforms.ToTensor()
])

# Class names
class_names = ['building', 'flooded', 'forest', 'mountains', 'sea', 'street']
@app.route('/')
def home():
    return render_template('main.html')

@app.route('/alert', methods=['POST'])
def alert():
    # Logic to send alert (you can integrate an API or email here if needed)
    alert_sent = True  # Simulating successful alert send
    return render_template('main.html', alert_sent=alert_sent)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    # Read and transform the image
    img = Image.open(BytesIO(file.read()))
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img)
    
    # Get the predicted class index
    _, predicted_class = output.max(1)
    predicted_class = predicted_class.item()  # Convert tensor to integer

    # Get the class name
    predicted_class_name = class_names[predicted_class]

    # If the class is 'flooded', show disclaimer and button
    if predicted_class_name == 'flooded':
        return render_template('main.html', prediction=predicted_class_name, show_disclaimer=True)

    # Otherwise, just show the predicted class
    return render_template('main.html', prediction=predicted_class_name, show_disclaimer=False)


    


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)




