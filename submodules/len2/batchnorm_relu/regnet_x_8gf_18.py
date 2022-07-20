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
        self.batchnorm2d29 = BatchNorm2d(720, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu26 = ReLU(inplace=True)

    def forward(self, x93):
        x94=self.batchnorm2d29(x93)
        x95=self.relu26(x94)
        return x95

m = M().eval()
x93 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x93)
end = time.time()
print(end-start)
