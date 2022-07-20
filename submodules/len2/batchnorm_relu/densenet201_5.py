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
        self.batchnorm2d5 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5 = ReLU(inplace=True)

    def forward(self, x19):
        x20=self.batchnorm2d5(x19)
        x21=self.relu5(x20)
        return x21

m = M().eval()
x19 = torch.randn(torch.Size([1, 128, 56, 56]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
