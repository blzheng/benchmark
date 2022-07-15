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
        self.batchnorm2d31 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x160):
        x161=self.batchnorm2d31(x160)
        return x161

m = M().eval()
x160 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x160)
end = time.time()
print(end-start)
