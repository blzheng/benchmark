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
        self.batchnorm2d32 = BatchNorm2d(416, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)

    def forward(self, x115):
        x116=self.batchnorm2d32(x115)
        x117=self.relu32(x116)
        return x117

m = M().eval()
x115 = torch.randn(torch.Size([1, 416, 28, 28]))
start = time.time()
output = m(x115)
end = time.time()
print(end-start)
