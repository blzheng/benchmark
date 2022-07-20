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
        self.batchnorm2d13 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu11 = ReLU(inplace=True)

    def forward(self, x44):
        x45=self.batchnorm2d13(x44)
        x46=self.relu11(x45)
        return x46

m = M().eval()
x44 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
