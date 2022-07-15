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
        self.batchnorm2d114 = BatchNorm2d(344, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x574):
        x575=self.batchnorm2d114(x574)
        return x575

m = M().eval()
x574 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x574)
end = time.time()
print(end-start)
