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
        self.batchnorm2d23 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x115, x102):
        x116=self.batchnorm2d23(x115)
        x117=operator.add(x116, x102)
        return x117

m = M().eval()
x115 = torch.randn(torch.Size([1, 80, 14, 14]))
x102 = torch.randn(torch.Size([1, 80, 14, 14]))
start = time.time()
output = m(x115, x102)
end = time.time()
print(end-start)
