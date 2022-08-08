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
        self.batchnorm2d46 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU()

    def forward(self, x184):
        x185=self.batchnorm2d46(x184)
        x186=self.relu19(x185)
        return x186

m = M().eval()
x184 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x184)
end = time.time()
print(end-start)
