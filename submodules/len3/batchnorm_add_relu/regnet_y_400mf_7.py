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
        self.batchnorm2d18 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu20 = ReLU(inplace=True)

    def forward(self, x86, x73):
        x87=self.batchnorm2d18(x86)
        x88=operator.add(x73, x87)
        x89=self.relu20(x88)
        return x89

m = M().eval()
x86 = torch.randn(torch.Size([1, 208, 14, 14]))
x73 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x86, x73)
end = time.time()
print(end-start)
