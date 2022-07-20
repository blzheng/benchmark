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
        self.batchnorm2d71 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x245):
        x246=self.batchnorm2d71(x245)
        x247=torch.nn.functional.relu(x246,inplace=True)
        return x247

m = M().eval()
x245 = torch.randn(torch.Size([1, 320, 5, 5]))
start = time.time()
output = m(x245)
end = time.time()
print(end-start)
