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
        self.batchnorm2d93 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x320):
        x321=self.batchnorm2d93(x320)
        x322=torch.nn.functional.relu(x321,inplace=True)
        return x322

m = M().eval()
x320 = torch.randn(torch.Size([1, 192, 5, 5]))
start = time.time()
output = m(x320)
end = time.time()
print(end-start)
