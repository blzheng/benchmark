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
        self.batchnorm2d104 = BatchNorm2d(640, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x370):
        x371=self.batchnorm2d104(x370)
        return x371

m = M().eval()
x370 = torch.randn(torch.Size([1, 640, 7, 7]))
start = time.time()
output = m(x370)
end = time.time()
print(end-start)
