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
        self.conv2d0 = Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.batchnorm2d0 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU6(inplace=True)
        self.conv2d1 = Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.batchnorm2d1 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu61 = ReLU6(inplace=True)
        self.conv2d2 = Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d2 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d3 = Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d3 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu62 = ReLU6(inplace=True)
        self.conv2d4 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d4 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu63 = ReLU6(inplace=True)
        self.conv2d5 = Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d5 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d6 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d6 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU6(inplace=True)
        self.conv2d7 = Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d7 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU6(inplace=True)
        self.conv2d8 = Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d8 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d9 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d9 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu66 = ReLU6(inplace=True)
        self.conv2d10 = Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d10 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU6(inplace=True)
        self.conv2d11 = Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d11 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d12 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d12 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu68 = ReLU6(inplace=True)
        self.conv2d13 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d13 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU6(inplace=True)
        self.conv2d14 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d14 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d15 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu610 = ReLU6(inplace=True)
        self.conv2d16 = Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d16 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu611 = ReLU6(inplace=True)
        self.conv2d17 = Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d18 = Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d18 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu612 = ReLU6(inplace=True)
        self.conv2d19 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        self.batchnorm2d19 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu613 = ReLU6(inplace=True)
        self.conv2d20 = Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d20 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d21 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d21 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu614 = ReLU6(inplace=True)
        self.conv2d22 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d22 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu615 = ReLU6(inplace=True)
        self.conv2d23 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d23 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d24 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu616 = ReLU6(inplace=True)
        self.conv2d25 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d25 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu617 = ReLU6(inplace=True)
        self.conv2d26 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d26 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d27 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d27 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu618 = ReLU6(inplace=True)
        self.conv2d28 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d28 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu619 = ReLU6(inplace=True)
        self.conv2d29 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d29 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d30 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu620 = ReLU6(inplace=True)
        self.conv2d31 = Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.batchnorm2d31 = BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu621 = ReLU6(inplace=True)
        self.conv2d32 = Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d32 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d33 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d33 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu622 = ReLU6(inplace=True)
        self.conv2d34 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d34 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu623 = ReLU6(inplace=True)
        self.conv2d35 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d36 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu624 = ReLU6(inplace=True)
        self.conv2d37 = Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d37 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu625 = ReLU6(inplace=True)
        self.conv2d38 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d39 = Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d39 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu626 = ReLU6(inplace=True)
        self.conv2d40 = Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
        self.batchnorm2d40 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu627 = ReLU6(inplace=True)
        self.conv2d41 = Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d41 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d42 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d42 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu628 = ReLU6(inplace=True)
        self.conv2d43 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d43 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu629 = ReLU6(inplace=True)
        self.conv2d44 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d44 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d45 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d45 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu630 = ReLU6(inplace=True)
        self.conv2d46 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d46 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu631 = ReLU6(inplace=True)
        self.conv2d47 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d47 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d48 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d48 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu632 = ReLU6(inplace=True)
        self.conv2d49 = Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.batchnorm2d49 = BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu633 = ReLU6(inplace=True)
        self.conv2d50 = Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d51 = Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu634 = ReLU6(inplace=True)
        self.dropout0 = Dropout(p=0.2, inplace=False)
        self.linear0 = Linear(in_features=1280, out_features=1000, bias=True)

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
        x1=self.conv2d0(x0)
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
        x2=self.batchnorm2d0(x1)
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
        x3=self.relu60(x2)
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
        x4=self.conv2d1(x3)
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
        x5=self.batchnorm2d1(x4)
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
        x6=self.relu61(x5)
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
        x7=self.conv2d2(x6)
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
        x8=self.batchnorm2d2(x7)
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
        x9=self.conv2d3(x8)
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
        x10=self.batchnorm2d3(x9)
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
        x11=self.relu62(x10)
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
        x12=self.conv2d4(x11)
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
        x13=self.batchnorm2d4(x12)
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
        x14=self.relu63(x13)
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
        x15=self.conv2d5(x14)
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
        x16=self.batchnorm2d5(x15)
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
        x17=self.conv2d6(x16)
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
        x18=self.batchnorm2d6(x17)
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
        x19=self.relu64(x18)
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
        x20=self.conv2d7(x19)
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
        x21=self.batchnorm2d7(x20)
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
        x22=self.relu65(x21)
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
        x23=self.conv2d8(x22)
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
        x24=self.batchnorm2d8(x23)
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
        x25=operator.add(x16, x24)
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
        x26=self.conv2d9(x25)
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
        x27=self.batchnorm2d9(x26)
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
        x28=self.relu66(x27)
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
        x29=self.conv2d10(x28)
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
        x30=self.batchnorm2d10(x29)
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
        x31=self.relu67(x30)
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
        x32=self.conv2d11(x31)
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
        x33=self.batchnorm2d11(x32)
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
        x34=self.conv2d12(x33)
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
        x35=self.batchnorm2d12(x34)
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
        x36=self.relu68(x35)
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
        x37=self.conv2d13(x36)
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
        x38=self.batchnorm2d13(x37)
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
        x39=self.relu69(x38)
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
        x40=self.conv2d14(x39)
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
        x41=self.batchnorm2d14(x40)
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
        x42=operator.add(x33, x41)
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
        x43=self.conv2d15(x42)
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
        x44=self.batchnorm2d15(x43)
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
        x45=self.relu610(x44)
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
        x46=self.conv2d16(x45)
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
        x47=self.batchnorm2d16(x46)
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
        x48=self.relu611(x47)
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
        x49=self.conv2d17(x48)
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
        x50=self.batchnorm2d17(x49)
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
        x51=operator.add(x42, x50)
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
        x52=self.conv2d18(x51)
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
        x53=self.batchnorm2d18(x52)
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
        x54=self.relu612(x53)
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
        x55=self.conv2d19(x54)
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
        x56=self.batchnorm2d19(x55)
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
        x57=self.relu613(x56)
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
        x58=self.conv2d20(x57)
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
        x59=self.batchnorm2d20(x58)
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
        x60=self.conv2d21(x59)
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
        x61=self.batchnorm2d21(x60)
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
        x62=self.relu614(x61)
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
        x63=self.conv2d22(x62)
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
        x64=self.batchnorm2d22(x63)
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
        x65=self.relu615(x64)
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
        x66=self.conv2d23(x65)
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
        x67=self.batchnorm2d23(x66)
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
        x68=operator.add(x59, x67)
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
        x69=self.conv2d24(x68)
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
        x70=self.batchnorm2d24(x69)
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
        x71=self.relu616(x70)
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
        x72=self.conv2d25(x71)
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
        x73=self.batchnorm2d25(x72)
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
        x74=self.relu617(x73)
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
        x75=self.conv2d26(x74)
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
        x76=self.batchnorm2d26(x75)
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
        x77=operator.add(x68, x76)
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
        x78=self.conv2d27(x77)
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
        x79=self.batchnorm2d27(x78)
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
        x80=self.relu618(x79)
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
        x81=self.conv2d28(x80)
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
        x82=self.batchnorm2d28(x81)
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
        x83=self.relu619(x82)
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
        x84=self.conv2d29(x83)
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
        x85=self.batchnorm2d29(x84)
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
        x86=operator.add(x77, x85)
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
        x87=self.conv2d30(x86)
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
        x88=self.batchnorm2d30(x87)
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
        x89=self.relu620(x88)
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
        x90=self.conv2d31(x89)
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
        x91=self.batchnorm2d31(x90)
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
        x92=self.relu621(x91)
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
        x93=self.conv2d32(x92)
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
        x94=self.batchnorm2d32(x93)
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
        x95=self.conv2d33(x94)
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
        x96=self.batchnorm2d33(x95)
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
        x97=self.relu622(x96)
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
        x98=self.conv2d34(x97)
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
        x99=self.batchnorm2d34(x98)
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
        x100=self.relu623(x99)
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
        x101=self.conv2d35(x100)
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
        x102=self.batchnorm2d35(x101)
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
        x103=operator.add(x94, x102)
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
        x104=self.conv2d36(x103)
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
        x105=self.batchnorm2d36(x104)
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
        x106=self.relu624(x105)
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
        x107=self.conv2d37(x106)
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
        x108=self.batchnorm2d37(x107)
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
        x109=self.relu625(x108)
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
        x110=self.conv2d38(x109)
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
        x111=self.batchnorm2d38(x110)
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
        x112=operator.add(x103, x111)
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
        x113=self.conv2d39(x112)
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
        x114=self.batchnorm2d39(x113)
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
        x115=self.relu626(x114)
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
        x116=self.conv2d40(x115)
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
        x117=self.batchnorm2d40(x116)
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
        x118=self.relu627(x117)
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
        x119=self.conv2d41(x118)
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
        x120=self.batchnorm2d41(x119)
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
        x121=self.conv2d42(x120)
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
        x122=self.batchnorm2d42(x121)
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
        x123=self.relu628(x122)
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
        x124=self.conv2d43(x123)
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
        x125=self.batchnorm2d43(x124)
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
        x126=self.relu629(x125)
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
        x127=self.conv2d44(x126)
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
        x128=self.batchnorm2d44(x127)
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
        x129=operator.add(x120, x128)
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
        x130=self.conv2d45(x129)
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
        x131=self.batchnorm2d45(x130)
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
        x132=self.relu630(x131)
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
        x133=self.conv2d46(x132)
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
        x134=self.batchnorm2d46(x133)
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
        x135=self.relu631(x134)
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
        x136=self.conv2d47(x135)
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
        x137=self.batchnorm2d47(x136)
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
        x138=operator.add(x129, x137)
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
        x139=self.conv2d48(x138)
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
        x140=self.batchnorm2d48(x139)
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
        x141=self.relu632(x140)
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
        x142=self.conv2d49(x141)
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
        x143=self.batchnorm2d49(x142)
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
        x144=self.relu633(x143)
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
        x145=self.conv2d50(x144)
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
        x146=self.batchnorm2d50(x145)
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
        x147=self.conv2d51(x146)
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
        x148=self.batchnorm2d51(x147)
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
        x149=self.relu634(x148)
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
        x150=torch.nn.functional.adaptive_avg_pool2d(x149, (1, 1))
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
        x151=torch.flatten(x150, 1)
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
        x152=self.dropout0(x151)
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
        x153=self.linear0(x152)
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

m = M().eval()
x = torch.rand(1, 3, 224, 224)
output = m(x)
