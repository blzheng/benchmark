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
        self.batchnorm2d75 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x387):
        x388=self.batchnorm2d75(x387)
        return x388

m = M().eval()
x387 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x387)
end = time.time()
print(end-start)
