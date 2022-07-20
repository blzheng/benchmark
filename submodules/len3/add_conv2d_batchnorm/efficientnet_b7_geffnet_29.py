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
        self.conv2d172 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d102 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x514, x500):
        x515=operator.add(x514, x500)
        x516=self.conv2d172(x515)
        x517=self.batchnorm2d102(x516)
        return x517

m = M().eval()
x514 = torch.randn(torch.Size([1, 224, 14, 14]))
x500 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x514, x500)
end = time.time()
print(end-start)
