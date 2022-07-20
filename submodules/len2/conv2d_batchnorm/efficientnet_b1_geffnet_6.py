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
        self.conv2d10 = Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.batchnorm2d6 = BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x31):
        x32=self.conv2d10(x31)
        x33=self.batchnorm2d6(x32)
        return x33

m = M().eval()
x31 = torch.randn(torch.Size([1, 96, 112, 112]))
start = time.time()
output = m(x31)
end = time.time()
print(end-start)
