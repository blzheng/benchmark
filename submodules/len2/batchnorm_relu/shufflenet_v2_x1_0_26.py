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
        self.batchnorm2d40 = BatchNorm2d(116, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)

    def forward(self, x259):
        x260=self.batchnorm2d40(x259)
        x261=self.relu26(x260)
        return x261

m = M().eval()
x259 = torch.randn(torch.Size([1, 116, 14, 14]))
start = time.time()
output = m(x259)
end = time.time()
print(end-start)
