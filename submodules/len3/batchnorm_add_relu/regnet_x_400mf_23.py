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
        self.batchnorm2d64 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)

    def forward(self, x208, x201):
        x209=self.batchnorm2d64(x208)
        x210=operator.add(x201, x209)
        x211=self.relu60(x210)
        return x211

m = M().eval()
x208 = torch.randn(torch.Size([1, 400, 7, 7]))
x201 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x208, x201)
end = time.time()
print(end-start)
