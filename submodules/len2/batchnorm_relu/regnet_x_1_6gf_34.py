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
        self.batchnorm2d54 = BatchNorm2d(912, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu50 = ReLU(inplace=True)

    def forward(self, x175):
        x176=self.batchnorm2d54(x175)
        x177=self.relu50(x176)
        return x177

m = M().eval()
x175 = torch.randn(torch.Size([1, 912, 7, 7]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)