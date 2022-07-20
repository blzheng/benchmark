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
        self.batchnorm2d80 = BatchNorm2d(888, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU(inplace=True)

    def forward(self, x412):
        x413=self.batchnorm2d80(x412)
        x414=self.relu101(x413)
        return x414

m = M().eval()
x412 = torch.randn(torch.Size([1, 888, 14, 14]))
start = time.time()
output = m(x412)
end = time.time()
print(end-start)
