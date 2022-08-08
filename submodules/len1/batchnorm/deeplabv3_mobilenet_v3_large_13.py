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
        self.batchnorm2d13 = BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x46):
        x47=self.batchnorm2d13(x46)
        return x47

m = M().eval()
x46 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x46)
end = time.time()
print(end-start)
