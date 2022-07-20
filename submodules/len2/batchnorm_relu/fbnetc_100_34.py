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
        self.batchnorm2d50 = BatchNorm2d(672, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.relu34 = ReLU(inplace=True)

    def forward(self, x163):
        x164=self.batchnorm2d50(x163)
        x165=self.relu34(x164)
        return x165

m = M().eval()
x163 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x163)
end = time.time()
print(end-start)