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
        self.batchnorm2d74 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x369):
        x370=self.batchnorm2d74(x369)
        return x370

m = M().eval()
x369 = torch.randn(torch.Size([1, 1632, 7, 7]))
start = time.time()
output = m(x369)
end = time.time()
print(end-start)
