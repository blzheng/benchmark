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
        self.conv2d43 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)
        self.batchnorm2d25 = BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x134):
        x135=self.conv2d43(x134)
        x136=self.batchnorm2d25(x135)
        return x136

m = M().eval()
x134 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x134)
end = time.time()
print(end-start)
