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
        self.batchnorm2d84 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)

    def forward(self, x279, x272):
        x280=self.batchnorm2d84(x279)
        x281=operator.add(x280, x272)
        x282=self.relu79(x281)
        return x282

m = M().eval()
x279 = torch.randn(torch.Size([1, 1024, 28, 28]))
x272 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x279, x272)
end = time.time()
print(end-start)
