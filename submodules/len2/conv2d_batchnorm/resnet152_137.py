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
        self.conv2d137 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d137 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x453):
        x454=self.conv2d137(x453)
        x455=self.batchnorm2d137(x454)
        return x455

m = M().eval()
x453 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x453)
end = time.time()
print(end-start)
