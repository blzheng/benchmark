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
        self.batchnorm2d93 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu88 = ReLU(inplace=True)

    def forward(self, x307, x300):
        x308=self.batchnorm2d93(x307)
        x309=operator.add(x308, x300)
        x310=self.relu88(x309)
        return x310

m = M().eval()
x307 = torch.randn(torch.Size([1, 1024, 14, 14]))
x300 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x307, x300)
end = time.time()
print(end-start)
