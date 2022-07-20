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
        self.batchnorm2d114 = BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu114 = ReLU(inplace=True)

    def forward(self, x405):
        x406=self.batchnorm2d114(x405)
        x407=self.relu114(x406)
        return x407

m = M().eval()
x405 = torch.randn(torch.Size([1, 928, 7, 7]))
start = time.time()
output = m(x405)
end = time.time()
print(end-start)
