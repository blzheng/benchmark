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
        self.batchnorm2d84 = BatchNorm2d(304, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x443):
        x444=self.batchnorm2d84(x443)
        return x444

m = M().eval()
x443 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x443)
end = time.time()
print(end-start)
