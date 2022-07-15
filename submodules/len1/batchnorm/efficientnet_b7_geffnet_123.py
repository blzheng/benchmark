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
        self.batchnorm2d123 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x620):
        x621=self.batchnorm2d123(x620)
        return x621

m = M().eval()
x620 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x620)
end = time.time()
print(end-start)
