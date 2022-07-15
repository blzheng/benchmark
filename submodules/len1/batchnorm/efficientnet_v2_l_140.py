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
        self.batchnorm2d140 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x684):
        x685=self.batchnorm2d140(x684)
        return x685

m = M().eval()
x684 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x684)
end = time.time()
print(end-start)
