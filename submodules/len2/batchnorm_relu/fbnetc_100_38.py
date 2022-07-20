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
        self.batchnorm2d56 = BatchNorm2d(1104, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu38 = ReLU(inplace=True)

    def forward(self, x182):
        x183=self.batchnorm2d56(x182)
        x184=self.relu38(x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 1104, 7, 7]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
