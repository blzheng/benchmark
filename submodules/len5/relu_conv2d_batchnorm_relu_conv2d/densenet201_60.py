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
        self.relu123 = ReLU(inplace=True)
        self.conv2d123 = Conv2d(1600, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d124 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu124 = ReLU(inplace=True)
        self.conv2d124 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x436):
        x437=self.relu123(x436)
        x438=self.conv2d123(x437)
        x439=self.batchnorm2d124(x438)
        x440=self.relu124(x439)
        x441=self.conv2d124(x440)
        return x441

m = M().eval()
x436 = torch.randn(torch.Size([1, 1600, 14, 14]))
start = time.time()
output = m(x436)
end = time.time()
print(end-start)
