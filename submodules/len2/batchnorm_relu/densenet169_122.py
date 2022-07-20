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
        self.batchnorm2d122 = BatchNorm2d(928, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu122 = ReLU(inplace=True)

    def forward(self, x433):
        x434=self.batchnorm2d122(x433)
        x435=self.relu122(x434)
        return x435

m = M().eval()
x433 = torch.randn(torch.Size([1, 928, 7, 7]))
start = time.time()
output = m(x433)
end = time.time()
print(end-start)
