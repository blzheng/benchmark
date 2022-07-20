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
        self.batchnorm2d13 = BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d24 = Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x70, x57):
        x71=self.batchnorm2d13(x70)
        x72=operator.add(x71, x57)
        x73=self.conv2d24(x72)
        return x73

m = M().eval()
x70 = torch.randn(torch.Size([1, 24, 56, 56]))
x57 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x70, x57)
end = time.time()
print(end-start)
