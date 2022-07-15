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
        self.batchnorm2d130 = BatchNorm2d(3456, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x651):
        x652=self.batchnorm2d130(x651)
        return x652

m = M().eval()
x651 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x651)
end = time.time()
print(end-start)
