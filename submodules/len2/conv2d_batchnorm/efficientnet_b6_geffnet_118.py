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
        self.conv2d198 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d118 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x591):
        x592=self.conv2d198(x591)
        x593=self.batchnorm2d118(x592)
        return x593

m = M().eval()
x591 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x591)
end = time.time()
print(end-start)
