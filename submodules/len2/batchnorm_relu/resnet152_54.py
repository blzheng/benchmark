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
        self.batchnorm2d83 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu79 = ReLU(inplace=True)

    def forward(self, x274):
        x275=self.batchnorm2d83(x274)
        x276=self.relu79(x275)
        return x276

m = M().eval()
x274 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x274)
end = time.time()
print(end-start)
