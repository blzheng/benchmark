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
        self.batchnorm2d79 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU(inplace=True)

    def forward(self, x410):
        x411=self.batchnorm2d79(x410)
        x412=self.relu101(x411)
        return x412

m = M().eval()
x410 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x410)
end = time.time()
print(end-start)
