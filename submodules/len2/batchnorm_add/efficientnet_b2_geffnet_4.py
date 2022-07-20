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
        self.batchnorm2d61 = BatchNorm2d(208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x306, x293):
        x307=self.batchnorm2d61(x306)
        x308=operator.add(x307, x293)
        return x308

m = M().eval()
x306 = torch.randn(torch.Size([1, 208, 7, 7]))
x293 = torch.randn(torch.Size([1, 208, 7, 7]))
start = time.time()
output = m(x306, x293)
end = time.time()
print(end-start)
