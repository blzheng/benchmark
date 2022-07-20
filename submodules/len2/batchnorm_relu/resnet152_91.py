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
        self.batchnorm2d139 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu136 = ReLU(inplace=True)

    def forward(self, x461):
        x462=self.batchnorm2d139(x461)
        x463=self.relu136(x462)
        return x463

m = M().eval()
x461 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x461)
end = time.time()
print(end-start)
