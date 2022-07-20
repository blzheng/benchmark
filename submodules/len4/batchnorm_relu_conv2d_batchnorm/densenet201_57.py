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
        self.batchnorm2d117 = BatchNorm2d(1504, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu117 = ReLU(inplace=True)
        self.conv2d117 = Conv2d(1504, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d118 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x414):
        x415=self.batchnorm2d117(x414)
        x416=self.relu117(x415)
        x417=self.conv2d117(x416)
        x418=self.batchnorm2d118(x417)
        return x418

m = M().eval()
x414 = torch.randn(torch.Size([1, 1504, 14, 14]))
start = time.time()
output = m(x414)
end = time.time()
print(end-start)
