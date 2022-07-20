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
        self.relu118 = ReLU(inplace=True)
        self.conv2d118 = Conv2d(864, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d119 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu119 = ReLU(inplace=True)

    def forward(self, x420):
        x421=self.relu118(x420)
        x422=self.conv2d118(x421)
        x423=self.batchnorm2d119(x422)
        x424=self.relu119(x423)
        return x424

m = M().eval()
x420 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x420)
end = time.time()
print(end-start)
