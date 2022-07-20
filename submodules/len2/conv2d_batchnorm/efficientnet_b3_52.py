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
        self.conv2d88 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d52 = BatchNorm2d(136, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x271):
        x272=self.conv2d88(x271)
        x273=self.batchnorm2d52(x272)
        return x273

m = M().eval()
x271 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x271)
end = time.time()
print(end-start)