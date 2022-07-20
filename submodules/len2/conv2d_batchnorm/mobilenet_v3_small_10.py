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
        self.conv2d12 = Conv2d(96, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=96, bias=False)
        self.batchnorm2d10 = BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x34):
        x35=self.conv2d12(x34)
        x36=self.batchnorm2d10(x35)
        return x36

m = M().eval()
x34 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)
