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
        self.batchnorm2d47 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu43 = ReLU(inplace=True)

    def forward(self, x154):
        x155=self.batchnorm2d47(x154)
        x156=self.relu43(x155)
        return x156

m = M().eval()
x154 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x154)
end = time.time()
print(end-start)
