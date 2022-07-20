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
        self.batchnorm2d52 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu65 = ReLU(inplace=True)

    def forward(self, x266):
        x267=self.batchnorm2d52(x266)
        x268=self.relu65(x267)
        return x268

m = M().eval()
x266 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x266)
end = time.time()
print(end-start)
