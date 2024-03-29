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
        self.batchnorm2d8 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)

    def forward(self, x24, x17):
        x25=self.batchnorm2d8(x24)
        x26=operator.add(x17, x25)
        x27=self.relu6(x26)
        return x27

m = M().eval()
x24 = torch.randn(torch.Size([1, 64, 28, 28]))
x17 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x24, x17)
end = time.time()
print(end-start)
