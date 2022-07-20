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
        self.batchnorm2d55 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu37 = ReLU(inplace=True)

    def forward(self, x179):
        x180=self.batchnorm2d55(x179)
        x181=self.relu37(x180)
        return x181

m = M().eval()
x179 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x179)
end = time.time()
print(end-start)
