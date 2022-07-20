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
        self.batchnorm2d27 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x103):
        x104=self.batchnorm2d27(x103)
        x105=torch.nn.functional.relu(x104,inplace=True)
        return x105

m = M().eval()
x103 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)