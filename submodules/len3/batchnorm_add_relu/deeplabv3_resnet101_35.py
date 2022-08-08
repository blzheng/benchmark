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
        self.batchnorm2d100 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu94 = ReLU(inplace=True)

    def forward(self, x331, x324):
        x332=self.batchnorm2d100(x331)
        x333=operator.add(x332, x324)
        x334=self.relu94(x333)
        return x334

m = M().eval()
x331 = torch.randn(torch.Size([1, 2048, 28, 28]))
x324 = torch.randn(torch.Size([1, 2048, 28, 28]))
start = time.time()
output = m(x331, x324)
end = time.time()
print(end-start)
