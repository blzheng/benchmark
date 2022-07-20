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
        self.batchnorm2d2 = BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d5 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)

    def forward(self, x13):
        x14=self.batchnorm2d2(x13)
        x15=self.conv2d5(x14)
        return x15

m = M().eval()
x13 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
