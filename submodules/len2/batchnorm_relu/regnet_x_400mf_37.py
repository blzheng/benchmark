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
        self.batchnorm2d59 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu55 = ReLU(inplace=True)

    def forward(self, x192):
        x193=self.batchnorm2d59(x192)
        x194=self.relu55(x193)
        return x194

m = M().eval()
x192 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x192)
end = time.time()
print(end-start)
