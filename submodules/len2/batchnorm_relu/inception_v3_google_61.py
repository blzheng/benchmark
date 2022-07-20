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
        self.batchnorm2d61 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x213):
        x214=self.batchnorm2d61(x213)
        x215=torch.nn.functional.relu(x214,inplace=True)
        return x215

m = M().eval()
x213 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x213)
end = time.time()
print(end-start)
