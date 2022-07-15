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
        self.batchnorm2d185 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x924):
        x925=self.batchnorm2d185(x924)
        return x925

m = M().eval()
x924 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x924)
end = time.time()
print(end-start)
