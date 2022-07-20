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
        self.batchnorm2d21 = BatchNorm2d(408, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu21 = ReLU(inplace=True)

    def forward(self, x68, x77):
        x69=self.batchnorm2d21(x68)
        x78=operator.add(x69, x77)
        x79=self.relu21(x78)
        return x79

m = M().eval()
x68 = torch.randn(torch.Size([1, 408, 14, 14]))
x77 = torch.randn(torch.Size([1, 408, 14, 14]))
start = time.time()
output = m(x68, x77)
end = time.time()
print(end-start)
