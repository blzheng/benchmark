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
        self.batchnorm2d138 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu133 = ReLU(inplace=True)

    def forward(self, x457, x450):
        x458=self.batchnorm2d138(x457)
        x459=operator.add(x458, x450)
        x460=self.relu133(x459)
        return x460

m = M().eval()
x457 = torch.randn(torch.Size([1, 1024, 14, 14]))
x450 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x457, x450)
end = time.time()
print(end-start)
