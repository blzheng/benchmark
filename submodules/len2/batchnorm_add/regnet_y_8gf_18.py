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
        self.batchnorm2d51 = BatchNorm2d(896, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x262, x249):
        x263=self.batchnorm2d51(x262)
        x264=operator.add(x249, x263)
        return x264

m = M().eval()
x262 = torch.randn(torch.Size([1, 896, 14, 14]))
x249 = torch.randn(torch.Size([1, 896, 14, 14]))
start = time.time()
output = m(x262, x249)
end = time.time()
print(end-start)
