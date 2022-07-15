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
        self.batchnorm2d41 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x208):
        x209=self.batchnorm2d41(x208)
        return x209

m = M().eval()
x208 = torch.randn(torch.Size([1, 432, 28, 28]))
start = time.time()
output = m(x208)
end = time.time()
print(end-start)
