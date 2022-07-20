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
        self.conv2d10 = Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        self.batchnorm2d10 = BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x28):
        x29=self.conv2d10(x28)
        x30=self.batchnorm2d10(x29)
        return x30

m = M().eval()
x28 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x28)
end = time.time()
print(end-start)
