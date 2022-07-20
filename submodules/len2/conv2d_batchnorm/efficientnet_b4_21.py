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
        self.conv2d35 = Conv2d(336, 336, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=336, bias=False)
        self.batchnorm2d21 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x106):
        x107=self.conv2d35(x106)
        x108=self.batchnorm2d21(x107)
        return x108

m = M().eval()
x106 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x106)
end = time.time()
print(end-start)