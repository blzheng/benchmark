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
        self.batchnorm2d98 = BatchNorm2d(672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu98 = ReLU(inplace=True)

    def forward(self, x349):
        x350=self.batchnorm2d98(x349)
        x351=self.relu98(x350)
        return x351

m = M().eval()
x349 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x349)
end = time.time()
print(end-start)
