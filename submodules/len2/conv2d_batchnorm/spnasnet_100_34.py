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
        self.conv2d34 = Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
        self.batchnorm2d34 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x110):
        x111=self.conv2d34(x110)
        x112=self.batchnorm2d34(x111)
        return x112

m = M().eval()
x110 = torch.randn(torch.Size([1, 240, 14, 14]))
start = time.time()
output = m(x110)
end = time.time()
print(end-start)
