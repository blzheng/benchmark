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
        self.batchnorm2d124 = BatchNorm2d(1344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu124 = ReLU(inplace=True)

    def forward(self, x440):
        x441=self.batchnorm2d124(x440)
        x442=self.relu124(x441)
        return x442

m = M().eval()
x440 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x440)
end = time.time()
print(end-start)
