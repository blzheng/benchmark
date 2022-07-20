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
        self.conv2d65 = Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
        self.batchnorm2d39 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x200):
        x201=self.conv2d65(x200)
        x202=self.batchnorm2d39(x201)
        return x202

m = M().eval()
x200 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x200)
end = time.time()
print(end-start)
