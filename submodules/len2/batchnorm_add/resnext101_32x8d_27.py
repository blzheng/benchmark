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
        self.batchnorm2d78 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x257, x250):
        x258=self.batchnorm2d78(x257)
        x259=operator.add(x258, x250)
        return x259

m = M().eval()
x257 = torch.randn(torch.Size([1, 1024, 14, 14]))
x250 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x257, x250)
end = time.time()
print(end-start)