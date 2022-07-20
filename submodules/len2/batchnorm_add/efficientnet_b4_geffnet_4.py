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
        self.batchnorm2d88 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x441, x428):
        x442=self.batchnorm2d88(x441)
        x443=operator.add(x442, x428)
        return x443

m = M().eval()
x441 = torch.randn(torch.Size([1, 272, 7, 7]))
x428 = torch.randn(torch.Size([1, 272, 7, 7]))
start = time.time()
output = m(x441, x428)
end = time.time()
print(end-start)
