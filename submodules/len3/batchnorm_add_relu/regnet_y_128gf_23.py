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
        self.batchnorm2d66 = BatchNorm2d(2904, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu84 = ReLU(inplace=True)

    def forward(self, x342, x329):
        x343=self.batchnorm2d66(x342)
        x344=operator.add(x329, x343)
        x345=self.relu84(x344)
        return x345

m = M().eval()
x342 = torch.randn(torch.Size([1, 2904, 14, 14]))
x329 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x342, x329)
end = time.time()
print(end-start)
