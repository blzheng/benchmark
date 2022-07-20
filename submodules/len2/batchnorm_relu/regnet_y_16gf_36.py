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
        self.batchnorm2d57 = BatchNorm2d(3024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)

    def forward(self, x287):
        x288=self.batchnorm2d57(x287)
        x289=self.relu70(x288)
        return x289

m = M().eval()
x287 = torch.randn(torch.Size([1, 3024, 7, 7]))
start = time.time()
output = m(x287)
end = time.time()
print(end-start)
