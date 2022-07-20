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
        self.batchnorm2d152 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu148 = ReLU(inplace=True)

    def forward(self, x503):
        x504=self.batchnorm2d152(x503)
        x505=self.relu148(x504)
        return x505

m = M().eval()
x503 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x503)
end = time.time()
print(end-start)
