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
        self.batchnorm2d209 = BatchNorm2d(640, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x1050):
        x1051=self.batchnorm2d209(x1050)
        return x1051

m = M().eval()
x1050 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x1050)
end = time.time()
print(end-start)
