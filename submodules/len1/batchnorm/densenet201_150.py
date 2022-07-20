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
        self.batchnorm2d150 = BatchNorm2d(1120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x531):
        x532=self.batchnorm2d150(x531)
        return x532

m = M().eval()
x531 = torch.randn(torch.Size([1, 1120, 7, 7]))
start = time.time()
output = m(x531)
end = time.time()
print(end-start)