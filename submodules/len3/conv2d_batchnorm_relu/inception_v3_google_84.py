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
        self.conv2d84 = Conv2d(1280, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x288):
        x289=self.conv2d84(x288)
        x290=self.batchnorm2d84(x289)
        x291=torch.nn.functional.relu(x290,inplace=True)
        return x291

m = M().eval()
x288 = torch.randn(torch.Size([1, 1280, 5, 5]))
start = time.time()
output = m(x288)
end = time.time()
print(end-start)
