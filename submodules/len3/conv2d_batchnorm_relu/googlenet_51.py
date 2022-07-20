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
        self.conv2d51 = Conv2d(832, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x186):
        x187=self.conv2d51(x186)
        x188=self.batchnorm2d51(x187)
        x189=torch.nn.functional.relu(x188,inplace=True)
        return x189

m = M().eval()
x186 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x186)
end = time.time()
print(end-start)
