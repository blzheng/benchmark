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
        self.batchnorm2d49 = BatchNorm2d(232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu32 = ReLU(inplace=True)

    def forward(self, x322):
        x323=self.batchnorm2d49(x322)
        x324=self.relu32(x323)
        return x324

m = M().eval()
x322 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x322)
end = time.time()
print(end-start)
