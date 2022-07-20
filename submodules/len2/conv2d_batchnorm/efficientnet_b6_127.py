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
        self.conv2d213 = Conv2d(576, 3456, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d127 = BatchNorm2d(3456, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x668):
        x669=self.conv2d213(x668)
        x670=self.batchnorm2d127(x669)
        return x670

m = M().eval()
x668 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x668)
end = time.time()
print(end-start)
