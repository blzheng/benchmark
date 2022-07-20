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
        self.batchnorm2d37 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x135):
        x136=self.batchnorm2d37(x135)
        x137=torch.nn.functional.relu(x136,inplace=True)
        return x137

m = M().eval()
x135 = torch.randn(torch.Size([1, 128, 12, 12]))
start = time.time()
output = m(x135)
end = time.time()
print(end-start)