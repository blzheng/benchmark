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
        self.conv2d17 = Conv2d(244, 244, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d17 = BatchNorm2d(244, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)

    def forward(self, x94):
        x100=self.conv2d17(x94)
        x101=self.batchnorm2d17(x100)
        x102=self.relu11(x101)
        return x102

m = M().eval()
x94 = torch.randn(torch.Size([1, 244, 28, 28]))
start = time.time()
output = m(x94)
end = time.time()
print(end-start)
