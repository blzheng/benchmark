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
        self.batchnorm2d10 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu7 = ReLU(inplace=True)

    def forward(self, x35, x28):
        x36=self.batchnorm2d10(x35)
        x37=operator.add(x36, x28)
        x38=self.relu7(x37)
        return x38

m = M().eval()
x35 = torch.randn(torch.Size([1, 256, 56, 56]))
x28 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x35, x28)
end = time.time()
print(end-start)
