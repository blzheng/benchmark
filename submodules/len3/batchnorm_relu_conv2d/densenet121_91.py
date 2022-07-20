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
        self.batchnorm2d92 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu92 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(576, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x328):
        x329=self.batchnorm2d92(x328)
        x330=self.relu92(x329)
        x331=self.conv2d92(x330)
        return x331

m = M().eval()
x328 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x328)
end = time.time()
print(end-start)