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
        self.batchnorm2d119 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu119 = ReLU(inplace=True)

    def forward(self, x422):
        x423=self.batchnorm2d119(x422)
        x424=self.relu119(x423)
        return x424

m = M().eval()
x422 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x422)
end = time.time()
print(end-start)
