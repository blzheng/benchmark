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
        self.batchnorm2d58 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x195):
        x196=self.batchnorm2d58(x195)
        return x196

m = M().eval()
x195 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x195)
end = time.time()
print(end-start)
