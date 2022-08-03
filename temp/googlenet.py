import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d0 = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.batchnorm2d0 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d0 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d1 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d1 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d2 = Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d1 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d3 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d4 = Conv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d4 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(96, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d6 = Conv2d(192, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d7 = Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d7 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d2 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d8 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d10 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d10 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d11 = Conv2d(128, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d12 = Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d13 = Conv2d(32, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d13 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d3 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d14 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d4 = MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d15 = Conv2d(480, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d16 = Conv2d(480, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d17 = Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(208, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d18 = Conv2d(480, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d19 = Conv2d(16, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d19 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d5 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d20 = Conv2d(480, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d21 = Conv2d(512, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d22 = Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d23 = Conv2d(112, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(224, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d25 = Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d25 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d6 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d26 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d27 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d28 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d28 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d29 = Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d30 = Conv2d(512, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(24, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d31 = Conv2d(24, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d31 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d7 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d32 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d33 = Conv2d(512, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(112, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d34 = Conv2d(512, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d34 = BatchNorm2d(144, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d35 = Conv2d(144, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(288, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(512, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d37 = Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d37 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d8 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d38 = Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(528, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d40 = Conv2d(528, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d41 = Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(528, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d43 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d43 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d9 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d44 = Conv2d(528, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d10 = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.conv2d45 = Conv2d(832, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d46 = Conv2d(832, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(160, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d47 = Conv2d(160, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d49 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d49 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d11 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d50 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d52 = Conv2d(832, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d53 = Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d53 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(832, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d54 = BatchNorm2d(48, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d55 = Conv2d(48, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.maxpool2d12 = MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)
        self.conv2d56 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d56 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.adaptiveavgpool2d0 = AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout0 = Dropout(p=0.2, inplace=False)
        self.linear0 = Linear(in_features=1024, out_features=1000, bias=True)

    def forward(self, x):
        x0=x
        if x0 is None:
            print('x0: {}'.format(x0))
        elif isinstance(x0, torch.Tensor):
            print('x0: {}'.format(x0.shape))
        elif isinstance(x0, tuple):
            tuple_shapes = '('
            for item in x0:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x0: {}'.format(tuple_shapes))
        else:
            print('x0: {}'.format(x0))
        x1=operator.getitem(x0, (slice(None, None, None), 0))
        if x1 is None:
            print('x1: {}'.format(x1))
        elif isinstance(x1, torch.Tensor):
            print('x1: {}'.format(x1.shape))
        elif isinstance(x1, tuple):
            tuple_shapes = '('
            for item in x1:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x1: {}'.format(tuple_shapes))
        else:
            print('x1: {}'.format(x1))
        x2=torch.unsqueeze(x1, 1)
        if x2 is None:
            print('x2: {}'.format(x2))
        elif isinstance(x2, torch.Tensor):
            print('x2: {}'.format(x2.shape))
        elif isinstance(x2, tuple):
            tuple_shapes = '('
            for item in x2:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x2: {}'.format(tuple_shapes))
        else:
            print('x2: {}'.format(x2))
        x3=operator.mul(x2, 0.458)
        if x3 is None:
            print('x3: {}'.format(x3))
        elif isinstance(x3, torch.Tensor):
            print('x3: {}'.format(x3.shape))
        elif isinstance(x3, tuple):
            tuple_shapes = '('
            for item in x3:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x3: {}'.format(tuple_shapes))
        else:
            print('x3: {}'.format(x3))
        x4=operator.add(x3, -0.030000000000000027)
        if x4 is None:
            print('x4: {}'.format(x4))
        elif isinstance(x4, torch.Tensor):
            print('x4: {}'.format(x4.shape))
        elif isinstance(x4, tuple):
            tuple_shapes = '('
            for item in x4:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x4: {}'.format(tuple_shapes))
        else:
            print('x4: {}'.format(x4))
        x5=operator.getitem(x0, (slice(None, None, None), 1))
        if x5 is None:
            print('x5: {}'.format(x5))
        elif isinstance(x5, torch.Tensor):
            print('x5: {}'.format(x5.shape))
        elif isinstance(x5, tuple):
            tuple_shapes = '('
            for item in x5:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x5: {}'.format(tuple_shapes))
        else:
            print('x5: {}'.format(x5))
        x6=torch.unsqueeze(x5, 1)
        if x6 is None:
            print('x6: {}'.format(x6))
        elif isinstance(x6, torch.Tensor):
            print('x6: {}'.format(x6.shape))
        elif isinstance(x6, tuple):
            tuple_shapes = '('
            for item in x6:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x6: {}'.format(tuple_shapes))
        else:
            print('x6: {}'.format(x6))
        x7=operator.mul(x6, 0.448)
        if x7 is None:
            print('x7: {}'.format(x7))
        elif isinstance(x7, torch.Tensor):
            print('x7: {}'.format(x7.shape))
        elif isinstance(x7, tuple):
            tuple_shapes = '('
            for item in x7:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x7: {}'.format(tuple_shapes))
        else:
            print('x7: {}'.format(x7))
        x8=operator.add(x7, -0.08799999999999997)
        if x8 is None:
            print('x8: {}'.format(x8))
        elif isinstance(x8, torch.Tensor):
            print('x8: {}'.format(x8.shape))
        elif isinstance(x8, tuple):
            tuple_shapes = '('
            for item in x8:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x8: {}'.format(tuple_shapes))
        else:
            print('x8: {}'.format(x8))
        x9=operator.getitem(x0, (slice(None, None, None), 2))
        if x9 is None:
            print('x9: {}'.format(x9))
        elif isinstance(x9, torch.Tensor):
            print('x9: {}'.format(x9.shape))
        elif isinstance(x9, tuple):
            tuple_shapes = '('
            for item in x9:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x9: {}'.format(tuple_shapes))
        else:
            print('x9: {}'.format(x9))
        x10=torch.unsqueeze(x9, 1)
        if x10 is None:
            print('x10: {}'.format(x10))
        elif isinstance(x10, torch.Tensor):
            print('x10: {}'.format(x10.shape))
        elif isinstance(x10, tuple):
            tuple_shapes = '('
            for item in x10:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x10: {}'.format(tuple_shapes))
        else:
            print('x10: {}'.format(x10))
        x11=operator.mul(x10, 0.45)
        if x11 is None:
            print('x11: {}'.format(x11))
        elif isinstance(x11, torch.Tensor):
            print('x11: {}'.format(x11.shape))
        elif isinstance(x11, tuple):
            tuple_shapes = '('
            for item in x11:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x11: {}'.format(tuple_shapes))
        else:
            print('x11: {}'.format(x11))
        x12=operator.add(x11, -0.18799999999999994)
        if x12 is None:
            print('x12: {}'.format(x12))
        elif isinstance(x12, torch.Tensor):
            print('x12: {}'.format(x12.shape))
        elif isinstance(x12, tuple):
            tuple_shapes = '('
            for item in x12:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x12: {}'.format(tuple_shapes))
        else:
            print('x12: {}'.format(x12))
        x13=torch.cat((x4, x8, x12), 1)
        if x13 is None:
            print('x13: {}'.format(x13))
        elif isinstance(x13, torch.Tensor):
            print('x13: {}'.format(x13.shape))
        elif isinstance(x13, tuple):
            tuple_shapes = '('
            for item in x13:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x13: {}'.format(tuple_shapes))
        else:
            print('x13: {}'.format(x13))
        x14=self.conv2d0(x13)
        if x14 is None:
            print('x14: {}'.format(x14))
        elif isinstance(x14, torch.Tensor):
            print('x14: {}'.format(x14.shape))
        elif isinstance(x14, tuple):
            tuple_shapes = '('
            for item in x14:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x14: {}'.format(tuple_shapes))
        else:
            print('x14: {}'.format(x14))
        x15=self.batchnorm2d0(x14)
        if x15 is None:
            print('x15: {}'.format(x15))
        elif isinstance(x15, torch.Tensor):
            print('x15: {}'.format(x15.shape))
        elif isinstance(x15, tuple):
            tuple_shapes = '('
            for item in x15:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x15: {}'.format(tuple_shapes))
        else:
            print('x15: {}'.format(x15))
        x16=torch.nn.functional.relu(x15,inplace=True)
        if x16 is None:
            print('x16: {}'.format(x16))
        elif isinstance(x16, torch.Tensor):
            print('x16: {}'.format(x16.shape))
        elif isinstance(x16, tuple):
            tuple_shapes = '('
            for item in x16:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x16: {}'.format(tuple_shapes))
        else:
            print('x16: {}'.format(x16))
        x17=self.maxpool2d0(x16)
        if x17 is None:
            print('x17: {}'.format(x17))
        elif isinstance(x17, torch.Tensor):
            print('x17: {}'.format(x17.shape))
        elif isinstance(x17, tuple):
            tuple_shapes = '('
            for item in x17:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x17: {}'.format(tuple_shapes))
        else:
            print('x17: {}'.format(x17))
        x18=self.conv2d1(x17)
        if x18 is None:
            print('x18: {}'.format(x18))
        elif isinstance(x18, torch.Tensor):
            print('x18: {}'.format(x18.shape))
        elif isinstance(x18, tuple):
            tuple_shapes = '('
            for item in x18:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x18: {}'.format(tuple_shapes))
        else:
            print('x18: {}'.format(x18))
        x19=self.batchnorm2d1(x18)
        if x19 is None:
            print('x19: {}'.format(x19))
        elif isinstance(x19, torch.Tensor):
            print('x19: {}'.format(x19.shape))
        elif isinstance(x19, tuple):
            tuple_shapes = '('
            for item in x19:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x19: {}'.format(tuple_shapes))
        else:
            print('x19: {}'.format(x19))
        x20=torch.nn.functional.relu(x19,inplace=True)
        if x20 is None:
            print('x20: {}'.format(x20))
        elif isinstance(x20, torch.Tensor):
            print('x20: {}'.format(x20.shape))
        elif isinstance(x20, tuple):
            tuple_shapes = '('
            for item in x20:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x20: {}'.format(tuple_shapes))
        else:
            print('x20: {}'.format(x20))
        x21=self.conv2d2(x20)
        if x21 is None:
            print('x21: {}'.format(x21))
        elif isinstance(x21, torch.Tensor):
            print('x21: {}'.format(x21.shape))
        elif isinstance(x21, tuple):
            tuple_shapes = '('
            for item in x21:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x21: {}'.format(tuple_shapes))
        else:
            print('x21: {}'.format(x21))
        x22=self.batchnorm2d2(x21)
        if x22 is None:
            print('x22: {}'.format(x22))
        elif isinstance(x22, torch.Tensor):
            print('x22: {}'.format(x22.shape))
        elif isinstance(x22, tuple):
            tuple_shapes = '('
            for item in x22:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x22: {}'.format(tuple_shapes))
        else:
            print('x22: {}'.format(x22))
        x23=torch.nn.functional.relu(x22,inplace=True)
        if x23 is None:
            print('x23: {}'.format(x23))
        elif isinstance(x23, torch.Tensor):
            print('x23: {}'.format(x23.shape))
        elif isinstance(x23, tuple):
            tuple_shapes = '('
            for item in x23:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x23: {}'.format(tuple_shapes))
        else:
            print('x23: {}'.format(x23))
        x24=self.maxpool2d1(x23)
        if x24 is None:
            print('x24: {}'.format(x24))
        elif isinstance(x24, torch.Tensor):
            print('x24: {}'.format(x24.shape))
        elif isinstance(x24, tuple):
            tuple_shapes = '('
            for item in x24:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x24: {}'.format(tuple_shapes))
        else:
            print('x24: {}'.format(x24))
        x25=self.conv2d3(x24)
        if x25 is None:
            print('x25: {}'.format(x25))
        elif isinstance(x25, torch.Tensor):
            print('x25: {}'.format(x25.shape))
        elif isinstance(x25, tuple):
            tuple_shapes = '('
            for item in x25:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x25: {}'.format(tuple_shapes))
        else:
            print('x25: {}'.format(x25))
        x26=self.batchnorm2d3(x25)
        if x26 is None:
            print('x26: {}'.format(x26))
        elif isinstance(x26, torch.Tensor):
            print('x26: {}'.format(x26.shape))
        elif isinstance(x26, tuple):
            tuple_shapes = '('
            for item in x26:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x26: {}'.format(tuple_shapes))
        else:
            print('x26: {}'.format(x26))
        x27=torch.nn.functional.relu(x26,inplace=True)
        if x27 is None:
            print('x27: {}'.format(x27))
        elif isinstance(x27, torch.Tensor):
            print('x27: {}'.format(x27.shape))
        elif isinstance(x27, tuple):
            tuple_shapes = '('
            for item in x27:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x27: {}'.format(tuple_shapes))
        else:
            print('x27: {}'.format(x27))
        x28=self.conv2d4(x24)
        if x28 is None:
            print('x28: {}'.format(x28))
        elif isinstance(x28, torch.Tensor):
            print('x28: {}'.format(x28.shape))
        elif isinstance(x28, tuple):
            tuple_shapes = '('
            for item in x28:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x28: {}'.format(tuple_shapes))
        else:
            print('x28: {}'.format(x28))
        x29=self.batchnorm2d4(x28)
        if x29 is None:
            print('x29: {}'.format(x29))
        elif isinstance(x29, torch.Tensor):
            print('x29: {}'.format(x29.shape))
        elif isinstance(x29, tuple):
            tuple_shapes = '('
            for item in x29:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x29: {}'.format(tuple_shapes))
        else:
            print('x29: {}'.format(x29))
        x30=torch.nn.functional.relu(x29,inplace=True)
        if x30 is None:
            print('x30: {}'.format(x30))
        elif isinstance(x30, torch.Tensor):
            print('x30: {}'.format(x30.shape))
        elif isinstance(x30, tuple):
            tuple_shapes = '('
            for item in x30:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x30: {}'.format(tuple_shapes))
        else:
            print('x30: {}'.format(x30))
        x31=self.conv2d5(x30)
        if x31 is None:
            print('x31: {}'.format(x31))
        elif isinstance(x31, torch.Tensor):
            print('x31: {}'.format(x31.shape))
        elif isinstance(x31, tuple):
            tuple_shapes = '('
            for item in x31:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x31: {}'.format(tuple_shapes))
        else:
            print('x31: {}'.format(x31))
        x32=self.batchnorm2d5(x31)
        if x32 is None:
            print('x32: {}'.format(x32))
        elif isinstance(x32, torch.Tensor):
            print('x32: {}'.format(x32.shape))
        elif isinstance(x32, tuple):
            tuple_shapes = '('
            for item in x32:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x32: {}'.format(tuple_shapes))
        else:
            print('x32: {}'.format(x32))
        x33=torch.nn.functional.relu(x32,inplace=True)
        if x33 is None:
            print('x33: {}'.format(x33))
        elif isinstance(x33, torch.Tensor):
            print('x33: {}'.format(x33.shape))
        elif isinstance(x33, tuple):
            tuple_shapes = '('
            for item in x33:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x33: {}'.format(tuple_shapes))
        else:
            print('x33: {}'.format(x33))
        x34=self.conv2d6(x24)
        if x34 is None:
            print('x34: {}'.format(x34))
        elif isinstance(x34, torch.Tensor):
            print('x34: {}'.format(x34.shape))
        elif isinstance(x34, tuple):
            tuple_shapes = '('
            for item in x34:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x34: {}'.format(tuple_shapes))
        else:
            print('x34: {}'.format(x34))
        x35=self.batchnorm2d6(x34)
        if x35 is None:
            print('x35: {}'.format(x35))
        elif isinstance(x35, torch.Tensor):
            print('x35: {}'.format(x35.shape))
        elif isinstance(x35, tuple):
            tuple_shapes = '('
            for item in x35:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x35: {}'.format(tuple_shapes))
        else:
            print('x35: {}'.format(x35))
        x36=torch.nn.functional.relu(x35,inplace=True)
        if x36 is None:
            print('x36: {}'.format(x36))
        elif isinstance(x36, torch.Tensor):
            print('x36: {}'.format(x36.shape))
        elif isinstance(x36, tuple):
            tuple_shapes = '('
            for item in x36:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x36: {}'.format(tuple_shapes))
        else:
            print('x36: {}'.format(x36))
        x37=self.conv2d7(x36)
        if x37 is None:
            print('x37: {}'.format(x37))
        elif isinstance(x37, torch.Tensor):
            print('x37: {}'.format(x37.shape))
        elif isinstance(x37, tuple):
            tuple_shapes = '('
            for item in x37:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x37: {}'.format(tuple_shapes))
        else:
            print('x37: {}'.format(x37))
        x38=self.batchnorm2d7(x37)
        if x38 is None:
            print('x38: {}'.format(x38))
        elif isinstance(x38, torch.Tensor):
            print('x38: {}'.format(x38.shape))
        elif isinstance(x38, tuple):
            tuple_shapes = '('
            for item in x38:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x38: {}'.format(tuple_shapes))
        else:
            print('x38: {}'.format(x38))
        x39=torch.nn.functional.relu(x38,inplace=True)
        if x39 is None:
            print('x39: {}'.format(x39))
        elif isinstance(x39, torch.Tensor):
            print('x39: {}'.format(x39.shape))
        elif isinstance(x39, tuple):
            tuple_shapes = '('
            for item in x39:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x39: {}'.format(tuple_shapes))
        else:
            print('x39: {}'.format(x39))
        x40=self.maxpool2d2(x24)
        if x40 is None:
            print('x40: {}'.format(x40))
        elif isinstance(x40, torch.Tensor):
            print('x40: {}'.format(x40.shape))
        elif isinstance(x40, tuple):
            tuple_shapes = '('
            for item in x40:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x40: {}'.format(tuple_shapes))
        else:
            print('x40: {}'.format(x40))
        x41=self.conv2d8(x40)
        if x41 is None:
            print('x41: {}'.format(x41))
        elif isinstance(x41, torch.Tensor):
            print('x41: {}'.format(x41.shape))
        elif isinstance(x41, tuple):
            tuple_shapes = '('
            for item in x41:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x41: {}'.format(tuple_shapes))
        else:
            print('x41: {}'.format(x41))
        x42=self.batchnorm2d8(x41)
        if x42 is None:
            print('x42: {}'.format(x42))
        elif isinstance(x42, torch.Tensor):
            print('x42: {}'.format(x42.shape))
        elif isinstance(x42, tuple):
            tuple_shapes = '('
            for item in x42:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x42: {}'.format(tuple_shapes))
        else:
            print('x42: {}'.format(x42))
        x43=torch.nn.functional.relu(x42,inplace=True)
        if x43 is None:
            print('x43: {}'.format(x43))
        elif isinstance(x43, torch.Tensor):
            print('x43: {}'.format(x43.shape))
        elif isinstance(x43, tuple):
            tuple_shapes = '('
            for item in x43:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x43: {}'.format(tuple_shapes))
        else:
            print('x43: {}'.format(x43))
        x44=torch.cat([x27, x33, x39, x43], 1)
        if x44 is None:
            print('x44: {}'.format(x44))
        elif isinstance(x44, torch.Tensor):
            print('x44: {}'.format(x44.shape))
        elif isinstance(x44, tuple):
            tuple_shapes = '('
            for item in x44:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x44: {}'.format(tuple_shapes))
        else:
            print('x44: {}'.format(x44))
        x45=self.conv2d9(x44)
        if x45 is None:
            print('x45: {}'.format(x45))
        elif isinstance(x45, torch.Tensor):
            print('x45: {}'.format(x45.shape))
        elif isinstance(x45, tuple):
            tuple_shapes = '('
            for item in x45:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x45: {}'.format(tuple_shapes))
        else:
            print('x45: {}'.format(x45))
        x46=self.batchnorm2d9(x45)
        if x46 is None:
            print('x46: {}'.format(x46))
        elif isinstance(x46, torch.Tensor):
            print('x46: {}'.format(x46.shape))
        elif isinstance(x46, tuple):
            tuple_shapes = '('
            for item in x46:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x46: {}'.format(tuple_shapes))
        else:
            print('x46: {}'.format(x46))
        x47=torch.nn.functional.relu(x46,inplace=True)
        if x47 is None:
            print('x47: {}'.format(x47))
        elif isinstance(x47, torch.Tensor):
            print('x47: {}'.format(x47.shape))
        elif isinstance(x47, tuple):
            tuple_shapes = '('
            for item in x47:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x47: {}'.format(tuple_shapes))
        else:
            print('x47: {}'.format(x47))
        x48=self.conv2d10(x44)
        if x48 is None:
            print('x48: {}'.format(x48))
        elif isinstance(x48, torch.Tensor):
            print('x48: {}'.format(x48.shape))
        elif isinstance(x48, tuple):
            tuple_shapes = '('
            for item in x48:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x48: {}'.format(tuple_shapes))
        else:
            print('x48: {}'.format(x48))
        x49=self.batchnorm2d10(x48)
        if x49 is None:
            print('x49: {}'.format(x49))
        elif isinstance(x49, torch.Tensor):
            print('x49: {}'.format(x49.shape))
        elif isinstance(x49, tuple):
            tuple_shapes = '('
            for item in x49:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x49: {}'.format(tuple_shapes))
        else:
            print('x49: {}'.format(x49))
        x50=torch.nn.functional.relu(x49,inplace=True)
        if x50 is None:
            print('x50: {}'.format(x50))
        elif isinstance(x50, torch.Tensor):
            print('x50: {}'.format(x50.shape))
        elif isinstance(x50, tuple):
            tuple_shapes = '('
            for item in x50:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x50: {}'.format(tuple_shapes))
        else:
            print('x50: {}'.format(x50))
        x51=self.conv2d11(x50)
        if x51 is None:
            print('x51: {}'.format(x51))
        elif isinstance(x51, torch.Tensor):
            print('x51: {}'.format(x51.shape))
        elif isinstance(x51, tuple):
            tuple_shapes = '('
            for item in x51:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x51: {}'.format(tuple_shapes))
        else:
            print('x51: {}'.format(x51))
        x52=self.batchnorm2d11(x51)
        if x52 is None:
            print('x52: {}'.format(x52))
        elif isinstance(x52, torch.Tensor):
            print('x52: {}'.format(x52.shape))
        elif isinstance(x52, tuple):
            tuple_shapes = '('
            for item in x52:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x52: {}'.format(tuple_shapes))
        else:
            print('x52: {}'.format(x52))
        x53=torch.nn.functional.relu(x52,inplace=True)
        if x53 is None:
            print('x53: {}'.format(x53))
        elif isinstance(x53, torch.Tensor):
            print('x53: {}'.format(x53.shape))
        elif isinstance(x53, tuple):
            tuple_shapes = '('
            for item in x53:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x53: {}'.format(tuple_shapes))
        else:
            print('x53: {}'.format(x53))
        x54=self.conv2d12(x44)
        if x54 is None:
            print('x54: {}'.format(x54))
        elif isinstance(x54, torch.Tensor):
            print('x54: {}'.format(x54.shape))
        elif isinstance(x54, tuple):
            tuple_shapes = '('
            for item in x54:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x54: {}'.format(tuple_shapes))
        else:
            print('x54: {}'.format(x54))
        x55=self.batchnorm2d12(x54)
        if x55 is None:
            print('x55: {}'.format(x55))
        elif isinstance(x55, torch.Tensor):
            print('x55: {}'.format(x55.shape))
        elif isinstance(x55, tuple):
            tuple_shapes = '('
            for item in x55:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x55: {}'.format(tuple_shapes))
        else:
            print('x55: {}'.format(x55))
        x56=torch.nn.functional.relu(x55,inplace=True)
        if x56 is None:
            print('x56: {}'.format(x56))
        elif isinstance(x56, torch.Tensor):
            print('x56: {}'.format(x56.shape))
        elif isinstance(x56, tuple):
            tuple_shapes = '('
            for item in x56:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x56: {}'.format(tuple_shapes))
        else:
            print('x56: {}'.format(x56))
        x57=self.conv2d13(x56)
        if x57 is None:
            print('x57: {}'.format(x57))
        elif isinstance(x57, torch.Tensor):
            print('x57: {}'.format(x57.shape))
        elif isinstance(x57, tuple):
            tuple_shapes = '('
            for item in x57:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x57: {}'.format(tuple_shapes))
        else:
            print('x57: {}'.format(x57))
        x58=self.batchnorm2d13(x57)
        if x58 is None:
            print('x58: {}'.format(x58))
        elif isinstance(x58, torch.Tensor):
            print('x58: {}'.format(x58.shape))
        elif isinstance(x58, tuple):
            tuple_shapes = '('
            for item in x58:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x58: {}'.format(tuple_shapes))
        else:
            print('x58: {}'.format(x58))
        x59=torch.nn.functional.relu(x58,inplace=True)
        if x59 is None:
            print('x59: {}'.format(x59))
        elif isinstance(x59, torch.Tensor):
            print('x59: {}'.format(x59.shape))
        elif isinstance(x59, tuple):
            tuple_shapes = '('
            for item in x59:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x59: {}'.format(tuple_shapes))
        else:
            print('x59: {}'.format(x59))
        x60=self.maxpool2d3(x44)
        if x60 is None:
            print('x60: {}'.format(x60))
        elif isinstance(x60, torch.Tensor):
            print('x60: {}'.format(x60.shape))
        elif isinstance(x60, tuple):
            tuple_shapes = '('
            for item in x60:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x60: {}'.format(tuple_shapes))
        else:
            print('x60: {}'.format(x60))
        x61=self.conv2d14(x60)
        if x61 is None:
            print('x61: {}'.format(x61))
        elif isinstance(x61, torch.Tensor):
            print('x61: {}'.format(x61.shape))
        elif isinstance(x61, tuple):
            tuple_shapes = '('
            for item in x61:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x61: {}'.format(tuple_shapes))
        else:
            print('x61: {}'.format(x61))
        x62=self.batchnorm2d14(x61)
        if x62 is None:
            print('x62: {}'.format(x62))
        elif isinstance(x62, torch.Tensor):
            print('x62: {}'.format(x62.shape))
        elif isinstance(x62, tuple):
            tuple_shapes = '('
            for item in x62:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x62: {}'.format(tuple_shapes))
        else:
            print('x62: {}'.format(x62))
        x63=torch.nn.functional.relu(x62,inplace=True)
        if x63 is None:
            print('x63: {}'.format(x63))
        elif isinstance(x63, torch.Tensor):
            print('x63: {}'.format(x63.shape))
        elif isinstance(x63, tuple):
            tuple_shapes = '('
            for item in x63:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x63: {}'.format(tuple_shapes))
        else:
            print('x63: {}'.format(x63))
        x64=torch.cat([x47, x53, x59, x63], 1)
        if x64 is None:
            print('x64: {}'.format(x64))
        elif isinstance(x64, torch.Tensor):
            print('x64: {}'.format(x64.shape))
        elif isinstance(x64, tuple):
            tuple_shapes = '('
            for item in x64:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x64: {}'.format(tuple_shapes))
        else:
            print('x64: {}'.format(x64))
        x65=self.maxpool2d4(x64)
        if x65 is None:
            print('x65: {}'.format(x65))
        elif isinstance(x65, torch.Tensor):
            print('x65: {}'.format(x65.shape))
        elif isinstance(x65, tuple):
            tuple_shapes = '('
            for item in x65:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x65: {}'.format(tuple_shapes))
        else:
            print('x65: {}'.format(x65))
        x66=self.conv2d15(x65)
        if x66 is None:
            print('x66: {}'.format(x66))
        elif isinstance(x66, torch.Tensor):
            print('x66: {}'.format(x66.shape))
        elif isinstance(x66, tuple):
            tuple_shapes = '('
            for item in x66:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x66: {}'.format(tuple_shapes))
        else:
            print('x66: {}'.format(x66))
        x67=self.batchnorm2d15(x66)
        if x67 is None:
            print('x67: {}'.format(x67))
        elif isinstance(x67, torch.Tensor):
            print('x67: {}'.format(x67.shape))
        elif isinstance(x67, tuple):
            tuple_shapes = '('
            for item in x67:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x67: {}'.format(tuple_shapes))
        else:
            print('x67: {}'.format(x67))
        x68=torch.nn.functional.relu(x67,inplace=True)
        if x68 is None:
            print('x68: {}'.format(x68))
        elif isinstance(x68, torch.Tensor):
            print('x68: {}'.format(x68.shape))
        elif isinstance(x68, tuple):
            tuple_shapes = '('
            for item in x68:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x68: {}'.format(tuple_shapes))
        else:
            print('x68: {}'.format(x68))
        x69=self.conv2d16(x65)
        if x69 is None:
            print('x69: {}'.format(x69))
        elif isinstance(x69, torch.Tensor):
            print('x69: {}'.format(x69.shape))
        elif isinstance(x69, tuple):
            tuple_shapes = '('
            for item in x69:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x69: {}'.format(tuple_shapes))
        else:
            print('x69: {}'.format(x69))
        x70=self.batchnorm2d16(x69)
        if x70 is None:
            print('x70: {}'.format(x70))
        elif isinstance(x70, torch.Tensor):
            print('x70: {}'.format(x70.shape))
        elif isinstance(x70, tuple):
            tuple_shapes = '('
            for item in x70:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x70: {}'.format(tuple_shapes))
        else:
            print('x70: {}'.format(x70))
        x71=torch.nn.functional.relu(x70,inplace=True)
        if x71 is None:
            print('x71: {}'.format(x71))
        elif isinstance(x71, torch.Tensor):
            print('x71: {}'.format(x71.shape))
        elif isinstance(x71, tuple):
            tuple_shapes = '('
            for item in x71:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x71: {}'.format(tuple_shapes))
        else:
            print('x71: {}'.format(x71))
        x72=self.conv2d17(x71)
        if x72 is None:
            print('x72: {}'.format(x72))
        elif isinstance(x72, torch.Tensor):
            print('x72: {}'.format(x72.shape))
        elif isinstance(x72, tuple):
            tuple_shapes = '('
            for item in x72:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x72: {}'.format(tuple_shapes))
        else:
            print('x72: {}'.format(x72))
        x73=self.batchnorm2d17(x72)
        if x73 is None:
            print('x73: {}'.format(x73))
        elif isinstance(x73, torch.Tensor):
            print('x73: {}'.format(x73.shape))
        elif isinstance(x73, tuple):
            tuple_shapes = '('
            for item in x73:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x73: {}'.format(tuple_shapes))
        else:
            print('x73: {}'.format(x73))
        x74=torch.nn.functional.relu(x73,inplace=True)
        if x74 is None:
            print('x74: {}'.format(x74))
        elif isinstance(x74, torch.Tensor):
            print('x74: {}'.format(x74.shape))
        elif isinstance(x74, tuple):
            tuple_shapes = '('
            for item in x74:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x74: {}'.format(tuple_shapes))
        else:
            print('x74: {}'.format(x74))
        x75=self.conv2d18(x65)
        if x75 is None:
            print('x75: {}'.format(x75))
        elif isinstance(x75, torch.Tensor):
            print('x75: {}'.format(x75.shape))
        elif isinstance(x75, tuple):
            tuple_shapes = '('
            for item in x75:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x75: {}'.format(tuple_shapes))
        else:
            print('x75: {}'.format(x75))
        x76=self.batchnorm2d18(x75)
        if x76 is None:
            print('x76: {}'.format(x76))
        elif isinstance(x76, torch.Tensor):
            print('x76: {}'.format(x76.shape))
        elif isinstance(x76, tuple):
            tuple_shapes = '('
            for item in x76:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x76: {}'.format(tuple_shapes))
        else:
            print('x76: {}'.format(x76))
        x77=torch.nn.functional.relu(x76,inplace=True)
        if x77 is None:
            print('x77: {}'.format(x77))
        elif isinstance(x77, torch.Tensor):
            print('x77: {}'.format(x77.shape))
        elif isinstance(x77, tuple):
            tuple_shapes = '('
            for item in x77:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x77: {}'.format(tuple_shapes))
        else:
            print('x77: {}'.format(x77))
        x78=self.conv2d19(x77)
        if x78 is None:
            print('x78: {}'.format(x78))
        elif isinstance(x78, torch.Tensor):
            print('x78: {}'.format(x78.shape))
        elif isinstance(x78, tuple):
            tuple_shapes = '('
            for item in x78:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x78: {}'.format(tuple_shapes))
        else:
            print('x78: {}'.format(x78))
        x79=self.batchnorm2d19(x78)
        if x79 is None:
            print('x79: {}'.format(x79))
        elif isinstance(x79, torch.Tensor):
            print('x79: {}'.format(x79.shape))
        elif isinstance(x79, tuple):
            tuple_shapes = '('
            for item in x79:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x79: {}'.format(tuple_shapes))
        else:
            print('x79: {}'.format(x79))
        x80=torch.nn.functional.relu(x79,inplace=True)
        if x80 is None:
            print('x80: {}'.format(x80))
        elif isinstance(x80, torch.Tensor):
            print('x80: {}'.format(x80.shape))
        elif isinstance(x80, tuple):
            tuple_shapes = '('
            for item in x80:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x80: {}'.format(tuple_shapes))
        else:
            print('x80: {}'.format(x80))
        x81=self.maxpool2d5(x65)
        if x81 is None:
            print('x81: {}'.format(x81))
        elif isinstance(x81, torch.Tensor):
            print('x81: {}'.format(x81.shape))
        elif isinstance(x81, tuple):
            tuple_shapes = '('
            for item in x81:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x81: {}'.format(tuple_shapes))
        else:
            print('x81: {}'.format(x81))
        x82=self.conv2d20(x81)
        if x82 is None:
            print('x82: {}'.format(x82))
        elif isinstance(x82, torch.Tensor):
            print('x82: {}'.format(x82.shape))
        elif isinstance(x82, tuple):
            tuple_shapes = '('
            for item in x82:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x82: {}'.format(tuple_shapes))
        else:
            print('x82: {}'.format(x82))
        x83=self.batchnorm2d20(x82)
        if x83 is None:
            print('x83: {}'.format(x83))
        elif isinstance(x83, torch.Tensor):
            print('x83: {}'.format(x83.shape))
        elif isinstance(x83, tuple):
            tuple_shapes = '('
            for item in x83:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x83: {}'.format(tuple_shapes))
        else:
            print('x83: {}'.format(x83))
        x84=torch.nn.functional.relu(x83,inplace=True)
        if x84 is None:
            print('x84: {}'.format(x84))
        elif isinstance(x84, torch.Tensor):
            print('x84: {}'.format(x84.shape))
        elif isinstance(x84, tuple):
            tuple_shapes = '('
            for item in x84:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x84: {}'.format(tuple_shapes))
        else:
            print('x84: {}'.format(x84))
        x85=torch.cat([x68, x74, x80, x84], 1)
        if x85 is None:
            print('x85: {}'.format(x85))
        elif isinstance(x85, torch.Tensor):
            print('x85: {}'.format(x85.shape))
        elif isinstance(x85, tuple):
            tuple_shapes = '('
            for item in x85:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x85: {}'.format(tuple_shapes))
        else:
            print('x85: {}'.format(x85))
        x86=self.conv2d21(x85)
        if x86 is None:
            print('x86: {}'.format(x86))
        elif isinstance(x86, torch.Tensor):
            print('x86: {}'.format(x86.shape))
        elif isinstance(x86, tuple):
            tuple_shapes = '('
            for item in x86:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x86: {}'.format(tuple_shapes))
        else:
            print('x86: {}'.format(x86))
        x87=self.batchnorm2d21(x86)
        if x87 is None:
            print('x87: {}'.format(x87))
        elif isinstance(x87, torch.Tensor):
            print('x87: {}'.format(x87.shape))
        elif isinstance(x87, tuple):
            tuple_shapes = '('
            for item in x87:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x87: {}'.format(tuple_shapes))
        else:
            print('x87: {}'.format(x87))
        x88=torch.nn.functional.relu(x87,inplace=True)
        if x88 is None:
            print('x88: {}'.format(x88))
        elif isinstance(x88, torch.Tensor):
            print('x88: {}'.format(x88.shape))
        elif isinstance(x88, tuple):
            tuple_shapes = '('
            for item in x88:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x88: {}'.format(tuple_shapes))
        else:
            print('x88: {}'.format(x88))
        x89=self.conv2d22(x85)
        if x89 is None:
            print('x89: {}'.format(x89))
        elif isinstance(x89, torch.Tensor):
            print('x89: {}'.format(x89.shape))
        elif isinstance(x89, tuple):
            tuple_shapes = '('
            for item in x89:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x89: {}'.format(tuple_shapes))
        else:
            print('x89: {}'.format(x89))
        x90=self.batchnorm2d22(x89)
        if x90 is None:
            print('x90: {}'.format(x90))
        elif isinstance(x90, torch.Tensor):
            print('x90: {}'.format(x90.shape))
        elif isinstance(x90, tuple):
            tuple_shapes = '('
            for item in x90:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x90: {}'.format(tuple_shapes))
        else:
            print('x90: {}'.format(x90))
        x91=torch.nn.functional.relu(x90,inplace=True)
        if x91 is None:
            print('x91: {}'.format(x91))
        elif isinstance(x91, torch.Tensor):
            print('x91: {}'.format(x91.shape))
        elif isinstance(x91, tuple):
            tuple_shapes = '('
            for item in x91:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x91: {}'.format(tuple_shapes))
        else:
            print('x91: {}'.format(x91))
        x92=self.conv2d23(x91)
        if x92 is None:
            print('x92: {}'.format(x92))
        elif isinstance(x92, torch.Tensor):
            print('x92: {}'.format(x92.shape))
        elif isinstance(x92, tuple):
            tuple_shapes = '('
            for item in x92:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x92: {}'.format(tuple_shapes))
        else:
            print('x92: {}'.format(x92))
        x93=self.batchnorm2d23(x92)
        if x93 is None:
            print('x93: {}'.format(x93))
        elif isinstance(x93, torch.Tensor):
            print('x93: {}'.format(x93.shape))
        elif isinstance(x93, tuple):
            tuple_shapes = '('
            for item in x93:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x93: {}'.format(tuple_shapes))
        else:
            print('x93: {}'.format(x93))
        x94=torch.nn.functional.relu(x93,inplace=True)
        if x94 is None:
            print('x94: {}'.format(x94))
        elif isinstance(x94, torch.Tensor):
            print('x94: {}'.format(x94.shape))
        elif isinstance(x94, tuple):
            tuple_shapes = '('
            for item in x94:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x94: {}'.format(tuple_shapes))
        else:
            print('x94: {}'.format(x94))
        x95=self.conv2d24(x85)
        if x95 is None:
            print('x95: {}'.format(x95))
        elif isinstance(x95, torch.Tensor):
            print('x95: {}'.format(x95.shape))
        elif isinstance(x95, tuple):
            tuple_shapes = '('
            for item in x95:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x95: {}'.format(tuple_shapes))
        else:
            print('x95: {}'.format(x95))
        x96=self.batchnorm2d24(x95)
        if x96 is None:
            print('x96: {}'.format(x96))
        elif isinstance(x96, torch.Tensor):
            print('x96: {}'.format(x96.shape))
        elif isinstance(x96, tuple):
            tuple_shapes = '('
            for item in x96:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x96: {}'.format(tuple_shapes))
        else:
            print('x96: {}'.format(x96))
        x97=torch.nn.functional.relu(x96,inplace=True)
        if x97 is None:
            print('x97: {}'.format(x97))
        elif isinstance(x97, torch.Tensor):
            print('x97: {}'.format(x97.shape))
        elif isinstance(x97, tuple):
            tuple_shapes = '('
            for item in x97:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x97: {}'.format(tuple_shapes))
        else:
            print('x97: {}'.format(x97))
        x98=self.conv2d25(x97)
        if x98 is None:
            print('x98: {}'.format(x98))
        elif isinstance(x98, torch.Tensor):
            print('x98: {}'.format(x98.shape))
        elif isinstance(x98, tuple):
            tuple_shapes = '('
            for item in x98:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x98: {}'.format(tuple_shapes))
        else:
            print('x98: {}'.format(x98))
        x99=self.batchnorm2d25(x98)
        if x99 is None:
            print('x99: {}'.format(x99))
        elif isinstance(x99, torch.Tensor):
            print('x99: {}'.format(x99.shape))
        elif isinstance(x99, tuple):
            tuple_shapes = '('
            for item in x99:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x99: {}'.format(tuple_shapes))
        else:
            print('x99: {}'.format(x99))
        x100=torch.nn.functional.relu(x99,inplace=True)
        if x100 is None:
            print('x100: {}'.format(x100))
        elif isinstance(x100, torch.Tensor):
            print('x100: {}'.format(x100.shape))
        elif isinstance(x100, tuple):
            tuple_shapes = '('
            for item in x100:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x100: {}'.format(tuple_shapes))
        else:
            print('x100: {}'.format(x100))
        x101=self.maxpool2d6(x85)
        if x101 is None:
            print('x101: {}'.format(x101))
        elif isinstance(x101, torch.Tensor):
            print('x101: {}'.format(x101.shape))
        elif isinstance(x101, tuple):
            tuple_shapes = '('
            for item in x101:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x101: {}'.format(tuple_shapes))
        else:
            print('x101: {}'.format(x101))
        x102=self.conv2d26(x101)
        if x102 is None:
            print('x102: {}'.format(x102))
        elif isinstance(x102, torch.Tensor):
            print('x102: {}'.format(x102.shape))
        elif isinstance(x102, tuple):
            tuple_shapes = '('
            for item in x102:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x102: {}'.format(tuple_shapes))
        else:
            print('x102: {}'.format(x102))
        x103=self.batchnorm2d26(x102)
        if x103 is None:
            print('x103: {}'.format(x103))
        elif isinstance(x103, torch.Tensor):
            print('x103: {}'.format(x103.shape))
        elif isinstance(x103, tuple):
            tuple_shapes = '('
            for item in x103:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x103: {}'.format(tuple_shapes))
        else:
            print('x103: {}'.format(x103))
        x104=torch.nn.functional.relu(x103,inplace=True)
        if x104 is None:
            print('x104: {}'.format(x104))
        elif isinstance(x104, torch.Tensor):
            print('x104: {}'.format(x104.shape))
        elif isinstance(x104, tuple):
            tuple_shapes = '('
            for item in x104:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x104: {}'.format(tuple_shapes))
        else:
            print('x104: {}'.format(x104))
        x105=torch.cat([x88, x94, x100, x104], 1)
        if x105 is None:
            print('x105: {}'.format(x105))
        elif isinstance(x105, torch.Tensor):
            print('x105: {}'.format(x105.shape))
        elif isinstance(x105, tuple):
            tuple_shapes = '('
            for item in x105:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x105: {}'.format(tuple_shapes))
        else:
            print('x105: {}'.format(x105))
        x106=self.conv2d27(x105)
        if x106 is None:
            print('x106: {}'.format(x106))
        elif isinstance(x106, torch.Tensor):
            print('x106: {}'.format(x106.shape))
        elif isinstance(x106, tuple):
            tuple_shapes = '('
            for item in x106:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x106: {}'.format(tuple_shapes))
        else:
            print('x106: {}'.format(x106))
        x107=self.batchnorm2d27(x106)
        if x107 is None:
            print('x107: {}'.format(x107))
        elif isinstance(x107, torch.Tensor):
            print('x107: {}'.format(x107.shape))
        elif isinstance(x107, tuple):
            tuple_shapes = '('
            for item in x107:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x107: {}'.format(tuple_shapes))
        else:
            print('x107: {}'.format(x107))
        x108=torch.nn.functional.relu(x107,inplace=True)
        if x108 is None:
            print('x108: {}'.format(x108))
        elif isinstance(x108, torch.Tensor):
            print('x108: {}'.format(x108.shape))
        elif isinstance(x108, tuple):
            tuple_shapes = '('
            for item in x108:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x108: {}'.format(tuple_shapes))
        else:
            print('x108: {}'.format(x108))
        x109=self.conv2d28(x105)
        if x109 is None:
            print('x109: {}'.format(x109))
        elif isinstance(x109, torch.Tensor):
            print('x109: {}'.format(x109.shape))
        elif isinstance(x109, tuple):
            tuple_shapes = '('
            for item in x109:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x109: {}'.format(tuple_shapes))
        else:
            print('x109: {}'.format(x109))
        x110=self.batchnorm2d28(x109)
        if x110 is None:
            print('x110: {}'.format(x110))
        elif isinstance(x110, torch.Tensor):
            print('x110: {}'.format(x110.shape))
        elif isinstance(x110, tuple):
            tuple_shapes = '('
            for item in x110:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x110: {}'.format(tuple_shapes))
        else:
            print('x110: {}'.format(x110))
        x111=torch.nn.functional.relu(x110,inplace=True)
        if x111 is None:
            print('x111: {}'.format(x111))
        elif isinstance(x111, torch.Tensor):
            print('x111: {}'.format(x111.shape))
        elif isinstance(x111, tuple):
            tuple_shapes = '('
            for item in x111:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x111: {}'.format(tuple_shapes))
        else:
            print('x111: {}'.format(x111))
        x112=self.conv2d29(x111)
        if x112 is None:
            print('x112: {}'.format(x112))
        elif isinstance(x112, torch.Tensor):
            print('x112: {}'.format(x112.shape))
        elif isinstance(x112, tuple):
            tuple_shapes = '('
            for item in x112:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x112: {}'.format(tuple_shapes))
        else:
            print('x112: {}'.format(x112))
        x113=self.batchnorm2d29(x112)
        if x113 is None:
            print('x113: {}'.format(x113))
        elif isinstance(x113, torch.Tensor):
            print('x113: {}'.format(x113.shape))
        elif isinstance(x113, tuple):
            tuple_shapes = '('
            for item in x113:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x113: {}'.format(tuple_shapes))
        else:
            print('x113: {}'.format(x113))
        x114=torch.nn.functional.relu(x113,inplace=True)
        if x114 is None:
            print('x114: {}'.format(x114))
        elif isinstance(x114, torch.Tensor):
            print('x114: {}'.format(x114.shape))
        elif isinstance(x114, tuple):
            tuple_shapes = '('
            for item in x114:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x114: {}'.format(tuple_shapes))
        else:
            print('x114: {}'.format(x114))
        x115=self.conv2d30(x105)
        if x115 is None:
            print('x115: {}'.format(x115))
        elif isinstance(x115, torch.Tensor):
            print('x115: {}'.format(x115.shape))
        elif isinstance(x115, tuple):
            tuple_shapes = '('
            for item in x115:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x115: {}'.format(tuple_shapes))
        else:
            print('x115: {}'.format(x115))
        x116=self.batchnorm2d30(x115)
        if x116 is None:
            print('x116: {}'.format(x116))
        elif isinstance(x116, torch.Tensor):
            print('x116: {}'.format(x116.shape))
        elif isinstance(x116, tuple):
            tuple_shapes = '('
            for item in x116:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x116: {}'.format(tuple_shapes))
        else:
            print('x116: {}'.format(x116))
        x117=torch.nn.functional.relu(x116,inplace=True)
        if x117 is None:
            print('x117: {}'.format(x117))
        elif isinstance(x117, torch.Tensor):
            print('x117: {}'.format(x117.shape))
        elif isinstance(x117, tuple):
            tuple_shapes = '('
            for item in x117:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x117: {}'.format(tuple_shapes))
        else:
            print('x117: {}'.format(x117))
        x118=self.conv2d31(x117)
        if x118 is None:
            print('x118: {}'.format(x118))
        elif isinstance(x118, torch.Tensor):
            print('x118: {}'.format(x118.shape))
        elif isinstance(x118, tuple):
            tuple_shapes = '('
            for item in x118:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x118: {}'.format(tuple_shapes))
        else:
            print('x118: {}'.format(x118))
        x119=self.batchnorm2d31(x118)
        if x119 is None:
            print('x119: {}'.format(x119))
        elif isinstance(x119, torch.Tensor):
            print('x119: {}'.format(x119.shape))
        elif isinstance(x119, tuple):
            tuple_shapes = '('
            for item in x119:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x119: {}'.format(tuple_shapes))
        else:
            print('x119: {}'.format(x119))
        x120=torch.nn.functional.relu(x119,inplace=True)
        if x120 is None:
            print('x120: {}'.format(x120))
        elif isinstance(x120, torch.Tensor):
            print('x120: {}'.format(x120.shape))
        elif isinstance(x120, tuple):
            tuple_shapes = '('
            for item in x120:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x120: {}'.format(tuple_shapes))
        else:
            print('x120: {}'.format(x120))
        x121=self.maxpool2d7(x105)
        if x121 is None:
            print('x121: {}'.format(x121))
        elif isinstance(x121, torch.Tensor):
            print('x121: {}'.format(x121.shape))
        elif isinstance(x121, tuple):
            tuple_shapes = '('
            for item in x121:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x121: {}'.format(tuple_shapes))
        else:
            print('x121: {}'.format(x121))
        x122=self.conv2d32(x121)
        if x122 is None:
            print('x122: {}'.format(x122))
        elif isinstance(x122, torch.Tensor):
            print('x122: {}'.format(x122.shape))
        elif isinstance(x122, tuple):
            tuple_shapes = '('
            for item in x122:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x122: {}'.format(tuple_shapes))
        else:
            print('x122: {}'.format(x122))
        x123=self.batchnorm2d32(x122)
        if x123 is None:
            print('x123: {}'.format(x123))
        elif isinstance(x123, torch.Tensor):
            print('x123: {}'.format(x123.shape))
        elif isinstance(x123, tuple):
            tuple_shapes = '('
            for item in x123:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x123: {}'.format(tuple_shapes))
        else:
            print('x123: {}'.format(x123))
        x124=torch.nn.functional.relu(x123,inplace=True)
        if x124 is None:
            print('x124: {}'.format(x124))
        elif isinstance(x124, torch.Tensor):
            print('x124: {}'.format(x124.shape))
        elif isinstance(x124, tuple):
            tuple_shapes = '('
            for item in x124:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x124: {}'.format(tuple_shapes))
        else:
            print('x124: {}'.format(x124))
        x125=torch.cat([x108, x114, x120, x124], 1)
        if x125 is None:
            print('x125: {}'.format(x125))
        elif isinstance(x125, torch.Tensor):
            print('x125: {}'.format(x125.shape))
        elif isinstance(x125, tuple):
            tuple_shapes = '('
            for item in x125:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x125: {}'.format(tuple_shapes))
        else:
            print('x125: {}'.format(x125))
        x126=self.conv2d33(x125)
        if x126 is None:
            print('x126: {}'.format(x126))
        elif isinstance(x126, torch.Tensor):
            print('x126: {}'.format(x126.shape))
        elif isinstance(x126, tuple):
            tuple_shapes = '('
            for item in x126:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x126: {}'.format(tuple_shapes))
        else:
            print('x126: {}'.format(x126))
        x127=self.batchnorm2d33(x126)
        if x127 is None:
            print('x127: {}'.format(x127))
        elif isinstance(x127, torch.Tensor):
            print('x127: {}'.format(x127.shape))
        elif isinstance(x127, tuple):
            tuple_shapes = '('
            for item in x127:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x127: {}'.format(tuple_shapes))
        else:
            print('x127: {}'.format(x127))
        x128=torch.nn.functional.relu(x127,inplace=True)
        if x128 is None:
            print('x128: {}'.format(x128))
        elif isinstance(x128, torch.Tensor):
            print('x128: {}'.format(x128.shape))
        elif isinstance(x128, tuple):
            tuple_shapes = '('
            for item in x128:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x128: {}'.format(tuple_shapes))
        else:
            print('x128: {}'.format(x128))
        x129=self.conv2d34(x125)
        if x129 is None:
            print('x129: {}'.format(x129))
        elif isinstance(x129, torch.Tensor):
            print('x129: {}'.format(x129.shape))
        elif isinstance(x129, tuple):
            tuple_shapes = '('
            for item in x129:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x129: {}'.format(tuple_shapes))
        else:
            print('x129: {}'.format(x129))
        x130=self.batchnorm2d34(x129)
        if x130 is None:
            print('x130: {}'.format(x130))
        elif isinstance(x130, torch.Tensor):
            print('x130: {}'.format(x130.shape))
        elif isinstance(x130, tuple):
            tuple_shapes = '('
            for item in x130:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x130: {}'.format(tuple_shapes))
        else:
            print('x130: {}'.format(x130))
        x131=torch.nn.functional.relu(x130,inplace=True)
        if x131 is None:
            print('x131: {}'.format(x131))
        elif isinstance(x131, torch.Tensor):
            print('x131: {}'.format(x131.shape))
        elif isinstance(x131, tuple):
            tuple_shapes = '('
            for item in x131:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x131: {}'.format(tuple_shapes))
        else:
            print('x131: {}'.format(x131))
        x132=self.conv2d35(x131)
        if x132 is None:
            print('x132: {}'.format(x132))
        elif isinstance(x132, torch.Tensor):
            print('x132: {}'.format(x132.shape))
        elif isinstance(x132, tuple):
            tuple_shapes = '('
            for item in x132:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x132: {}'.format(tuple_shapes))
        else:
            print('x132: {}'.format(x132))
        x133=self.batchnorm2d35(x132)
        if x133 is None:
            print('x133: {}'.format(x133))
        elif isinstance(x133, torch.Tensor):
            print('x133: {}'.format(x133.shape))
        elif isinstance(x133, tuple):
            tuple_shapes = '('
            for item in x133:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x133: {}'.format(tuple_shapes))
        else:
            print('x133: {}'.format(x133))
        x134=torch.nn.functional.relu(x133,inplace=True)
        if x134 is None:
            print('x134: {}'.format(x134))
        elif isinstance(x134, torch.Tensor):
            print('x134: {}'.format(x134.shape))
        elif isinstance(x134, tuple):
            tuple_shapes = '('
            for item in x134:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x134: {}'.format(tuple_shapes))
        else:
            print('x134: {}'.format(x134))
        x135=self.conv2d36(x125)
        if x135 is None:
            print('x135: {}'.format(x135))
        elif isinstance(x135, torch.Tensor):
            print('x135: {}'.format(x135.shape))
        elif isinstance(x135, tuple):
            tuple_shapes = '('
            for item in x135:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x135: {}'.format(tuple_shapes))
        else:
            print('x135: {}'.format(x135))
        x136=self.batchnorm2d36(x135)
        if x136 is None:
            print('x136: {}'.format(x136))
        elif isinstance(x136, torch.Tensor):
            print('x136: {}'.format(x136.shape))
        elif isinstance(x136, tuple):
            tuple_shapes = '('
            for item in x136:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x136: {}'.format(tuple_shapes))
        else:
            print('x136: {}'.format(x136))
        x137=torch.nn.functional.relu(x136,inplace=True)
        if x137 is None:
            print('x137: {}'.format(x137))
        elif isinstance(x137, torch.Tensor):
            print('x137: {}'.format(x137.shape))
        elif isinstance(x137, tuple):
            tuple_shapes = '('
            for item in x137:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x137: {}'.format(tuple_shapes))
        else:
            print('x137: {}'.format(x137))
        x138=self.conv2d37(x137)
        if x138 is None:
            print('x138: {}'.format(x138))
        elif isinstance(x138, torch.Tensor):
            print('x138: {}'.format(x138.shape))
        elif isinstance(x138, tuple):
            tuple_shapes = '('
            for item in x138:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x138: {}'.format(tuple_shapes))
        else:
            print('x138: {}'.format(x138))
        x139=self.batchnorm2d37(x138)
        if x139 is None:
            print('x139: {}'.format(x139))
        elif isinstance(x139, torch.Tensor):
            print('x139: {}'.format(x139.shape))
        elif isinstance(x139, tuple):
            tuple_shapes = '('
            for item in x139:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x139: {}'.format(tuple_shapes))
        else:
            print('x139: {}'.format(x139))
        x140=torch.nn.functional.relu(x139,inplace=True)
        if x140 is None:
            print('x140: {}'.format(x140))
        elif isinstance(x140, torch.Tensor):
            print('x140: {}'.format(x140.shape))
        elif isinstance(x140, tuple):
            tuple_shapes = '('
            for item in x140:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x140: {}'.format(tuple_shapes))
        else:
            print('x140: {}'.format(x140))
        x141=self.maxpool2d8(x125)
        if x141 is None:
            print('x141: {}'.format(x141))
        elif isinstance(x141, torch.Tensor):
            print('x141: {}'.format(x141.shape))
        elif isinstance(x141, tuple):
            tuple_shapes = '('
            for item in x141:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x141: {}'.format(tuple_shapes))
        else:
            print('x141: {}'.format(x141))
        x142=self.conv2d38(x141)
        if x142 is None:
            print('x142: {}'.format(x142))
        elif isinstance(x142, torch.Tensor):
            print('x142: {}'.format(x142.shape))
        elif isinstance(x142, tuple):
            tuple_shapes = '('
            for item in x142:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x142: {}'.format(tuple_shapes))
        else:
            print('x142: {}'.format(x142))
        x143=self.batchnorm2d38(x142)
        if x143 is None:
            print('x143: {}'.format(x143))
        elif isinstance(x143, torch.Tensor):
            print('x143: {}'.format(x143.shape))
        elif isinstance(x143, tuple):
            tuple_shapes = '('
            for item in x143:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x143: {}'.format(tuple_shapes))
        else:
            print('x143: {}'.format(x143))
        x144=torch.nn.functional.relu(x143,inplace=True)
        if x144 is None:
            print('x144: {}'.format(x144))
        elif isinstance(x144, torch.Tensor):
            print('x144: {}'.format(x144.shape))
        elif isinstance(x144, tuple):
            tuple_shapes = '('
            for item in x144:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x144: {}'.format(tuple_shapes))
        else:
            print('x144: {}'.format(x144))
        x145=torch.cat([x128, x134, x140, x144], 1)
        if x145 is None:
            print('x145: {}'.format(x145))
        elif isinstance(x145, torch.Tensor):
            print('x145: {}'.format(x145.shape))
        elif isinstance(x145, tuple):
            tuple_shapes = '('
            for item in x145:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x145: {}'.format(tuple_shapes))
        else:
            print('x145: {}'.format(x145))
        x146=self.conv2d39(x145)
        if x146 is None:
            print('x146: {}'.format(x146))
        elif isinstance(x146, torch.Tensor):
            print('x146: {}'.format(x146.shape))
        elif isinstance(x146, tuple):
            tuple_shapes = '('
            for item in x146:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x146: {}'.format(tuple_shapes))
        else:
            print('x146: {}'.format(x146))
        x147=self.batchnorm2d39(x146)
        if x147 is None:
            print('x147: {}'.format(x147))
        elif isinstance(x147, torch.Tensor):
            print('x147: {}'.format(x147.shape))
        elif isinstance(x147, tuple):
            tuple_shapes = '('
            for item in x147:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x147: {}'.format(tuple_shapes))
        else:
            print('x147: {}'.format(x147))
        x148=torch.nn.functional.relu(x147,inplace=True)
        if x148 is None:
            print('x148: {}'.format(x148))
        elif isinstance(x148, torch.Tensor):
            print('x148: {}'.format(x148.shape))
        elif isinstance(x148, tuple):
            tuple_shapes = '('
            for item in x148:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x148: {}'.format(tuple_shapes))
        else:
            print('x148: {}'.format(x148))
        x149=self.conv2d40(x145)
        if x149 is None:
            print('x149: {}'.format(x149))
        elif isinstance(x149, torch.Tensor):
            print('x149: {}'.format(x149.shape))
        elif isinstance(x149, tuple):
            tuple_shapes = '('
            for item in x149:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x149: {}'.format(tuple_shapes))
        else:
            print('x149: {}'.format(x149))
        x150=self.batchnorm2d40(x149)
        if x150 is None:
            print('x150: {}'.format(x150))
        elif isinstance(x150, torch.Tensor):
            print('x150: {}'.format(x150.shape))
        elif isinstance(x150, tuple):
            tuple_shapes = '('
            for item in x150:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x150: {}'.format(tuple_shapes))
        else:
            print('x150: {}'.format(x150))
        x151=torch.nn.functional.relu(x150,inplace=True)
        if x151 is None:
            print('x151: {}'.format(x151))
        elif isinstance(x151, torch.Tensor):
            print('x151: {}'.format(x151.shape))
        elif isinstance(x151, tuple):
            tuple_shapes = '('
            for item in x151:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x151: {}'.format(tuple_shapes))
        else:
            print('x151: {}'.format(x151))
        x152=self.conv2d41(x151)
        if x152 is None:
            print('x152: {}'.format(x152))
        elif isinstance(x152, torch.Tensor):
            print('x152: {}'.format(x152.shape))
        elif isinstance(x152, tuple):
            tuple_shapes = '('
            for item in x152:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x152: {}'.format(tuple_shapes))
        else:
            print('x152: {}'.format(x152))
        x153=self.batchnorm2d41(x152)
        if x153 is None:
            print('x153: {}'.format(x153))
        elif isinstance(x153, torch.Tensor):
            print('x153: {}'.format(x153.shape))
        elif isinstance(x153, tuple):
            tuple_shapes = '('
            for item in x153:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x153: {}'.format(tuple_shapes))
        else:
            print('x153: {}'.format(x153))
        x154=torch.nn.functional.relu(x153,inplace=True)
        if x154 is None:
            print('x154: {}'.format(x154))
        elif isinstance(x154, torch.Tensor):
            print('x154: {}'.format(x154.shape))
        elif isinstance(x154, tuple):
            tuple_shapes = '('
            for item in x154:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x154: {}'.format(tuple_shapes))
        else:
            print('x154: {}'.format(x154))
        x155=self.conv2d42(x145)
        if x155 is None:
            print('x155: {}'.format(x155))
        elif isinstance(x155, torch.Tensor):
            print('x155: {}'.format(x155.shape))
        elif isinstance(x155, tuple):
            tuple_shapes = '('
            for item in x155:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x155: {}'.format(tuple_shapes))
        else:
            print('x155: {}'.format(x155))
        x156=self.batchnorm2d42(x155)
        if x156 is None:
            print('x156: {}'.format(x156))
        elif isinstance(x156, torch.Tensor):
            print('x156: {}'.format(x156.shape))
        elif isinstance(x156, tuple):
            tuple_shapes = '('
            for item in x156:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x156: {}'.format(tuple_shapes))
        else:
            print('x156: {}'.format(x156))
        x157=torch.nn.functional.relu(x156,inplace=True)
        if x157 is None:
            print('x157: {}'.format(x157))
        elif isinstance(x157, torch.Tensor):
            print('x157: {}'.format(x157.shape))
        elif isinstance(x157, tuple):
            tuple_shapes = '('
            for item in x157:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x157: {}'.format(tuple_shapes))
        else:
            print('x157: {}'.format(x157))
        x158=self.conv2d43(x157)
        if x158 is None:
            print('x158: {}'.format(x158))
        elif isinstance(x158, torch.Tensor):
            print('x158: {}'.format(x158.shape))
        elif isinstance(x158, tuple):
            tuple_shapes = '('
            for item in x158:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x158: {}'.format(tuple_shapes))
        else:
            print('x158: {}'.format(x158))
        x159=self.batchnorm2d43(x158)
        if x159 is None:
            print('x159: {}'.format(x159))
        elif isinstance(x159, torch.Tensor):
            print('x159: {}'.format(x159.shape))
        elif isinstance(x159, tuple):
            tuple_shapes = '('
            for item in x159:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x159: {}'.format(tuple_shapes))
        else:
            print('x159: {}'.format(x159))
        x160=torch.nn.functional.relu(x159,inplace=True)
        if x160 is None:
            print('x160: {}'.format(x160))
        elif isinstance(x160, torch.Tensor):
            print('x160: {}'.format(x160.shape))
        elif isinstance(x160, tuple):
            tuple_shapes = '('
            for item in x160:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x160: {}'.format(tuple_shapes))
        else:
            print('x160: {}'.format(x160))
        x161=self.maxpool2d9(x145)
        if x161 is None:
            print('x161: {}'.format(x161))
        elif isinstance(x161, torch.Tensor):
            print('x161: {}'.format(x161.shape))
        elif isinstance(x161, tuple):
            tuple_shapes = '('
            for item in x161:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x161: {}'.format(tuple_shapes))
        else:
            print('x161: {}'.format(x161))
        x162=self.conv2d44(x161)
        if x162 is None:
            print('x162: {}'.format(x162))
        elif isinstance(x162, torch.Tensor):
            print('x162: {}'.format(x162.shape))
        elif isinstance(x162, tuple):
            tuple_shapes = '('
            for item in x162:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x162: {}'.format(tuple_shapes))
        else:
            print('x162: {}'.format(x162))
        x163=self.batchnorm2d44(x162)
        if x163 is None:
            print('x163: {}'.format(x163))
        elif isinstance(x163, torch.Tensor):
            print('x163: {}'.format(x163.shape))
        elif isinstance(x163, tuple):
            tuple_shapes = '('
            for item in x163:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x163: {}'.format(tuple_shapes))
        else:
            print('x163: {}'.format(x163))
        x164=torch.nn.functional.relu(x163,inplace=True)
        if x164 is None:
            print('x164: {}'.format(x164))
        elif isinstance(x164, torch.Tensor):
            print('x164: {}'.format(x164.shape))
        elif isinstance(x164, tuple):
            tuple_shapes = '('
            for item in x164:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x164: {}'.format(tuple_shapes))
        else:
            print('x164: {}'.format(x164))
        x165=torch.cat([x148, x154, x160, x164], 1)
        if x165 is None:
            print('x165: {}'.format(x165))
        elif isinstance(x165, torch.Tensor):
            print('x165: {}'.format(x165.shape))
        elif isinstance(x165, tuple):
            tuple_shapes = '('
            for item in x165:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x165: {}'.format(tuple_shapes))
        else:
            print('x165: {}'.format(x165))
        x166=self.maxpool2d10(x165)
        if x166 is None:
            print('x166: {}'.format(x166))
        elif isinstance(x166, torch.Tensor):
            print('x166: {}'.format(x166.shape))
        elif isinstance(x166, tuple):
            tuple_shapes = '('
            for item in x166:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x166: {}'.format(tuple_shapes))
        else:
            print('x166: {}'.format(x166))
        x167=self.conv2d45(x166)
        if x167 is None:
            print('x167: {}'.format(x167))
        elif isinstance(x167, torch.Tensor):
            print('x167: {}'.format(x167.shape))
        elif isinstance(x167, tuple):
            tuple_shapes = '('
            for item in x167:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x167: {}'.format(tuple_shapes))
        else:
            print('x167: {}'.format(x167))
        x168=self.batchnorm2d45(x167)
        if x168 is None:
            print('x168: {}'.format(x168))
        elif isinstance(x168, torch.Tensor):
            print('x168: {}'.format(x168.shape))
        elif isinstance(x168, tuple):
            tuple_shapes = '('
            for item in x168:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x168: {}'.format(tuple_shapes))
        else:
            print('x168: {}'.format(x168))
        x169=torch.nn.functional.relu(x168,inplace=True)
        if x169 is None:
            print('x169: {}'.format(x169))
        elif isinstance(x169, torch.Tensor):
            print('x169: {}'.format(x169.shape))
        elif isinstance(x169, tuple):
            tuple_shapes = '('
            for item in x169:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x169: {}'.format(tuple_shapes))
        else:
            print('x169: {}'.format(x169))
        x170=self.conv2d46(x166)
        if x170 is None:
            print('x170: {}'.format(x170))
        elif isinstance(x170, torch.Tensor):
            print('x170: {}'.format(x170.shape))
        elif isinstance(x170, tuple):
            tuple_shapes = '('
            for item in x170:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x170: {}'.format(tuple_shapes))
        else:
            print('x170: {}'.format(x170))
        x171=self.batchnorm2d46(x170)
        if x171 is None:
            print('x171: {}'.format(x171))
        elif isinstance(x171, torch.Tensor):
            print('x171: {}'.format(x171.shape))
        elif isinstance(x171, tuple):
            tuple_shapes = '('
            for item in x171:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x171: {}'.format(tuple_shapes))
        else:
            print('x171: {}'.format(x171))
        x172=torch.nn.functional.relu(x171,inplace=True)
        if x172 is None:
            print('x172: {}'.format(x172))
        elif isinstance(x172, torch.Tensor):
            print('x172: {}'.format(x172.shape))
        elif isinstance(x172, tuple):
            tuple_shapes = '('
            for item in x172:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x172: {}'.format(tuple_shapes))
        else:
            print('x172: {}'.format(x172))
        x173=self.conv2d47(x172)
        if x173 is None:
            print('x173: {}'.format(x173))
        elif isinstance(x173, torch.Tensor):
            print('x173: {}'.format(x173.shape))
        elif isinstance(x173, tuple):
            tuple_shapes = '('
            for item in x173:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x173: {}'.format(tuple_shapes))
        else:
            print('x173: {}'.format(x173))
        x174=self.batchnorm2d47(x173)
        if x174 is None:
            print('x174: {}'.format(x174))
        elif isinstance(x174, torch.Tensor):
            print('x174: {}'.format(x174.shape))
        elif isinstance(x174, tuple):
            tuple_shapes = '('
            for item in x174:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x174: {}'.format(tuple_shapes))
        else:
            print('x174: {}'.format(x174))
        x175=torch.nn.functional.relu(x174,inplace=True)
        if x175 is None:
            print('x175: {}'.format(x175))
        elif isinstance(x175, torch.Tensor):
            print('x175: {}'.format(x175.shape))
        elif isinstance(x175, tuple):
            tuple_shapes = '('
            for item in x175:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x175: {}'.format(tuple_shapes))
        else:
            print('x175: {}'.format(x175))
        x176=self.conv2d48(x166)
        if x176 is None:
            print('x176: {}'.format(x176))
        elif isinstance(x176, torch.Tensor):
            print('x176: {}'.format(x176.shape))
        elif isinstance(x176, tuple):
            tuple_shapes = '('
            for item in x176:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x176: {}'.format(tuple_shapes))
        else:
            print('x176: {}'.format(x176))
        x177=self.batchnorm2d48(x176)
        if x177 is None:
            print('x177: {}'.format(x177))
        elif isinstance(x177, torch.Tensor):
            print('x177: {}'.format(x177.shape))
        elif isinstance(x177, tuple):
            tuple_shapes = '('
            for item in x177:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x177: {}'.format(tuple_shapes))
        else:
            print('x177: {}'.format(x177))
        x178=torch.nn.functional.relu(x177,inplace=True)
        if x178 is None:
            print('x178: {}'.format(x178))
        elif isinstance(x178, torch.Tensor):
            print('x178: {}'.format(x178.shape))
        elif isinstance(x178, tuple):
            tuple_shapes = '('
            for item in x178:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x178: {}'.format(tuple_shapes))
        else:
            print('x178: {}'.format(x178))
        x179=self.conv2d49(x178)
        if x179 is None:
            print('x179: {}'.format(x179))
        elif isinstance(x179, torch.Tensor):
            print('x179: {}'.format(x179.shape))
        elif isinstance(x179, tuple):
            tuple_shapes = '('
            for item in x179:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x179: {}'.format(tuple_shapes))
        else:
            print('x179: {}'.format(x179))
        x180=self.batchnorm2d49(x179)
        if x180 is None:
            print('x180: {}'.format(x180))
        elif isinstance(x180, torch.Tensor):
            print('x180: {}'.format(x180.shape))
        elif isinstance(x180, tuple):
            tuple_shapes = '('
            for item in x180:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x180: {}'.format(tuple_shapes))
        else:
            print('x180: {}'.format(x180))
        x181=torch.nn.functional.relu(x180,inplace=True)
        if x181 is None:
            print('x181: {}'.format(x181))
        elif isinstance(x181, torch.Tensor):
            print('x181: {}'.format(x181.shape))
        elif isinstance(x181, tuple):
            tuple_shapes = '('
            for item in x181:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x181: {}'.format(tuple_shapes))
        else:
            print('x181: {}'.format(x181))
        x182=self.maxpool2d11(x166)
        if x182 is None:
            print('x182: {}'.format(x182))
        elif isinstance(x182, torch.Tensor):
            print('x182: {}'.format(x182.shape))
        elif isinstance(x182, tuple):
            tuple_shapes = '('
            for item in x182:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x182: {}'.format(tuple_shapes))
        else:
            print('x182: {}'.format(x182))
        x183=self.conv2d50(x182)
        if x183 is None:
            print('x183: {}'.format(x183))
        elif isinstance(x183, torch.Tensor):
            print('x183: {}'.format(x183.shape))
        elif isinstance(x183, tuple):
            tuple_shapes = '('
            for item in x183:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x183: {}'.format(tuple_shapes))
        else:
            print('x183: {}'.format(x183))
        x184=self.batchnorm2d50(x183)
        if x184 is None:
            print('x184: {}'.format(x184))
        elif isinstance(x184, torch.Tensor):
            print('x184: {}'.format(x184.shape))
        elif isinstance(x184, tuple):
            tuple_shapes = '('
            for item in x184:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x184: {}'.format(tuple_shapes))
        else:
            print('x184: {}'.format(x184))
        x185=torch.nn.functional.relu(x184,inplace=True)
        if x185 is None:
            print('x185: {}'.format(x185))
        elif isinstance(x185, torch.Tensor):
            print('x185: {}'.format(x185.shape))
        elif isinstance(x185, tuple):
            tuple_shapes = '('
            for item in x185:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x185: {}'.format(tuple_shapes))
        else:
            print('x185: {}'.format(x185))
        x186=torch.cat([x169, x175, x181, x185], 1)
        if x186 is None:
            print('x186: {}'.format(x186))
        elif isinstance(x186, torch.Tensor):
            print('x186: {}'.format(x186.shape))
        elif isinstance(x186, tuple):
            tuple_shapes = '('
            for item in x186:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x186: {}'.format(tuple_shapes))
        else:
            print('x186: {}'.format(x186))
        x187=self.conv2d51(x186)
        if x187 is None:
            print('x187: {}'.format(x187))
        elif isinstance(x187, torch.Tensor):
            print('x187: {}'.format(x187.shape))
        elif isinstance(x187, tuple):
            tuple_shapes = '('
            for item in x187:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x187: {}'.format(tuple_shapes))
        else:
            print('x187: {}'.format(x187))
        x188=self.batchnorm2d51(x187)
        if x188 is None:
            print('x188: {}'.format(x188))
        elif isinstance(x188, torch.Tensor):
            print('x188: {}'.format(x188.shape))
        elif isinstance(x188, tuple):
            tuple_shapes = '('
            for item in x188:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x188: {}'.format(tuple_shapes))
        else:
            print('x188: {}'.format(x188))
        x189=torch.nn.functional.relu(x188,inplace=True)
        if x189 is None:
            print('x189: {}'.format(x189))
        elif isinstance(x189, torch.Tensor):
            print('x189: {}'.format(x189.shape))
        elif isinstance(x189, tuple):
            tuple_shapes = '('
            for item in x189:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x189: {}'.format(tuple_shapes))
        else:
            print('x189: {}'.format(x189))
        x190=self.conv2d52(x186)
        if x190 is None:
            print('x190: {}'.format(x190))
        elif isinstance(x190, torch.Tensor):
            print('x190: {}'.format(x190.shape))
        elif isinstance(x190, tuple):
            tuple_shapes = '('
            for item in x190:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x190: {}'.format(tuple_shapes))
        else:
            print('x190: {}'.format(x190))
        x191=self.batchnorm2d52(x190)
        if x191 is None:
            print('x191: {}'.format(x191))
        elif isinstance(x191, torch.Tensor):
            print('x191: {}'.format(x191.shape))
        elif isinstance(x191, tuple):
            tuple_shapes = '('
            for item in x191:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x191: {}'.format(tuple_shapes))
        else:
            print('x191: {}'.format(x191))
        x192=torch.nn.functional.relu(x191,inplace=True)
        if x192 is None:
            print('x192: {}'.format(x192))
        elif isinstance(x192, torch.Tensor):
            print('x192: {}'.format(x192.shape))
        elif isinstance(x192, tuple):
            tuple_shapes = '('
            for item in x192:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x192: {}'.format(tuple_shapes))
        else:
            print('x192: {}'.format(x192))
        x193=self.conv2d53(x192)
        if x193 is None:
            print('x193: {}'.format(x193))
        elif isinstance(x193, torch.Tensor):
            print('x193: {}'.format(x193.shape))
        elif isinstance(x193, tuple):
            tuple_shapes = '('
            for item in x193:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x193: {}'.format(tuple_shapes))
        else:
            print('x193: {}'.format(x193))
        x194=self.batchnorm2d53(x193)
        if x194 is None:
            print('x194: {}'.format(x194))
        elif isinstance(x194, torch.Tensor):
            print('x194: {}'.format(x194.shape))
        elif isinstance(x194, tuple):
            tuple_shapes = '('
            for item in x194:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x194: {}'.format(tuple_shapes))
        else:
            print('x194: {}'.format(x194))
        x195=torch.nn.functional.relu(x194,inplace=True)
        if x195 is None:
            print('x195: {}'.format(x195))
        elif isinstance(x195, torch.Tensor):
            print('x195: {}'.format(x195.shape))
        elif isinstance(x195, tuple):
            tuple_shapes = '('
            for item in x195:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x195: {}'.format(tuple_shapes))
        else:
            print('x195: {}'.format(x195))
        x196=self.conv2d54(x186)
        if x196 is None:
            print('x196: {}'.format(x196))
        elif isinstance(x196, torch.Tensor):
            print('x196: {}'.format(x196.shape))
        elif isinstance(x196, tuple):
            tuple_shapes = '('
            for item in x196:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x196: {}'.format(tuple_shapes))
        else:
            print('x196: {}'.format(x196))
        x197=self.batchnorm2d54(x196)
        if x197 is None:
            print('x197: {}'.format(x197))
        elif isinstance(x197, torch.Tensor):
            print('x197: {}'.format(x197.shape))
        elif isinstance(x197, tuple):
            tuple_shapes = '('
            for item in x197:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x197: {}'.format(tuple_shapes))
        else:
            print('x197: {}'.format(x197))
        x198=torch.nn.functional.relu(x197,inplace=True)
        if x198 is None:
            print('x198: {}'.format(x198))
        elif isinstance(x198, torch.Tensor):
            print('x198: {}'.format(x198.shape))
        elif isinstance(x198, tuple):
            tuple_shapes = '('
            for item in x198:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x198: {}'.format(tuple_shapes))
        else:
            print('x198: {}'.format(x198))
        x199=self.conv2d55(x198)
        if x199 is None:
            print('x199: {}'.format(x199))
        elif isinstance(x199, torch.Tensor):
            print('x199: {}'.format(x199.shape))
        elif isinstance(x199, tuple):
            tuple_shapes = '('
            for item in x199:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x199: {}'.format(tuple_shapes))
        else:
            print('x199: {}'.format(x199))
        x200=self.batchnorm2d55(x199)
        if x200 is None:
            print('x200: {}'.format(x200))
        elif isinstance(x200, torch.Tensor):
            print('x200: {}'.format(x200.shape))
        elif isinstance(x200, tuple):
            tuple_shapes = '('
            for item in x200:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x200: {}'.format(tuple_shapes))
        else:
            print('x200: {}'.format(x200))
        x201=torch.nn.functional.relu(x200,inplace=True)
        if x201 is None:
            print('x201: {}'.format(x201))
        elif isinstance(x201, torch.Tensor):
            print('x201: {}'.format(x201.shape))
        elif isinstance(x201, tuple):
            tuple_shapes = '('
            for item in x201:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x201: {}'.format(tuple_shapes))
        else:
            print('x201: {}'.format(x201))
        x202=self.maxpool2d12(x186)
        if x202 is None:
            print('x202: {}'.format(x202))
        elif isinstance(x202, torch.Tensor):
            print('x202: {}'.format(x202.shape))
        elif isinstance(x202, tuple):
            tuple_shapes = '('
            for item in x202:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x202: {}'.format(tuple_shapes))
        else:
            print('x202: {}'.format(x202))
        x203=self.conv2d56(x202)
        if x203 is None:
            print('x203: {}'.format(x203))
        elif isinstance(x203, torch.Tensor):
            print('x203: {}'.format(x203.shape))
        elif isinstance(x203, tuple):
            tuple_shapes = '('
            for item in x203:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x203: {}'.format(tuple_shapes))
        else:
            print('x203: {}'.format(x203))
        x204=self.batchnorm2d56(x203)
        if x204 is None:
            print('x204: {}'.format(x204))
        elif isinstance(x204, torch.Tensor):
            print('x204: {}'.format(x204.shape))
        elif isinstance(x204, tuple):
            tuple_shapes = '('
            for item in x204:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x204: {}'.format(tuple_shapes))
        else:
            print('x204: {}'.format(x204))
        x205=torch.nn.functional.relu(x204,inplace=True)
        if x205 is None:
            print('x205: {}'.format(x205))
        elif isinstance(x205, torch.Tensor):
            print('x205: {}'.format(x205.shape))
        elif isinstance(x205, tuple):
            tuple_shapes = '('
            for item in x205:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x205: {}'.format(tuple_shapes))
        else:
            print('x205: {}'.format(x205))
        x206=torch.cat([x189, x195, x201, x205], 1)
        if x206 is None:
            print('x206: {}'.format(x206))
        elif isinstance(x206, torch.Tensor):
            print('x206: {}'.format(x206.shape))
        elif isinstance(x206, tuple):
            tuple_shapes = '('
            for item in x206:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x206: {}'.format(tuple_shapes))
        else:
            print('x206: {}'.format(x206))
        x207=self.adaptiveavgpool2d0(x206)
        if x207 is None:
            print('x207: {}'.format(x207))
        elif isinstance(x207, torch.Tensor):
            print('x207: {}'.format(x207.shape))
        elif isinstance(x207, tuple):
            tuple_shapes = '('
            for item in x207:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x207: {}'.format(tuple_shapes))
        else:
            print('x207: {}'.format(x207))
        x208=torch.flatten(x207, 1)
        if x208 is None:
            print('x208: {}'.format(x208))
        elif isinstance(x208, torch.Tensor):
            print('x208: {}'.format(x208.shape))
        elif isinstance(x208, tuple):
            tuple_shapes = '('
            for item in x208:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x208: {}'.format(tuple_shapes))
        else:
            print('x208: {}'.format(x208))
        x209=self.dropout0(x208)
        if x209 is None:
            print('x209: {}'.format(x209))
        elif isinstance(x209, torch.Tensor):
            print('x209: {}'.format(x209.shape))
        elif isinstance(x209, tuple):
            tuple_shapes = '('
            for item in x209:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x209: {}'.format(tuple_shapes))
        else:
            print('x209: {}'.format(x209))
        x210=self.linear0(x209)
        if x210 is None:
            print('x210: {}'.format(x210))
        elif isinstance(x210, torch.Tensor):
            print('x210: {}'.format(x210.shape))
        elif isinstance(x210, tuple):
            tuple_shapes = '('
            for item in x210:
               if isinstance(item, torch.Tensor):
                   tuple_shapes += str(item.shape) + ', '
               else:
                   tuple_shapes += str(item) + ', '
            tuple_shapes += ')'
            print('x210: {}'.format(tuple_shapes))
        else:
            print('x210: {}'.format(x210))

m = M().eval()
x = torch.rand(1, 3, 224, 224)
output = m(x)
