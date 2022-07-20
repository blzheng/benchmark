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
        self.conv2d76 = Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d76 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x261):
        x262=self.conv2d76(x261)
        x263=self.batchnorm2d76(x262)
        x264=torch.nn.functional.relu(x263,inplace=True)
        return x264

m = M().eval()
x261 = torch.randn(torch.Size([1, 1280, 5, 5]))
start = time.time()
output = m(x261)
end = time.time()
print(end-start)
