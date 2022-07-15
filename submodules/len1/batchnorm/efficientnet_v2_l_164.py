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
        self.batchnorm2d164 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x812):
        x813=self.batchnorm2d164(x812)
        return x813

m = M().eval()
x812 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x812)
end = time.time()
print(end-start)
