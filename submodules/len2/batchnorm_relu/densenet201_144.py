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
        self.batchnorm2d144 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu144 = ReLU(inplace=True)

    def forward(self, x510):
        x511=self.batchnorm2d144(x510)
        x512=self.relu144(x511)
        return x512

m = M().eval()
x510 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x510)
end = time.time()
print(end-start)
