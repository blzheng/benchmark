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
        self.batchnorm2d45 = BatchNorm2d(864, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)

    def forward(self, x130):
        x131=self.batchnorm2d45(x130)
        x132=self.relu30(x131)
        return x132

m = M().eval()
x130 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x130)
end = time.time()
print(end-start)
