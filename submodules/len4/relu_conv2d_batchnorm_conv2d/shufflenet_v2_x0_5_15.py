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
        self.relu34 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d53 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d54 = Conv2d(96, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x345):
        x346=self.relu34(x345)
        x347=self.conv2d53(x346)
        x348=self.batchnorm2d53(x347)
        x349=self.conv2d54(x348)
        return x349

m = M().eval()
x345 = torch.randn(torch.Size([1, 96, 7, 7]))
start = time.time()
output = m(x345)
end = time.time()
print(end-start)
