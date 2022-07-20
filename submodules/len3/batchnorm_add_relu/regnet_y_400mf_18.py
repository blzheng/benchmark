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
        self.batchnorm2d49 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)

    def forward(self, x248, x235):
        x249=self.batchnorm2d49(x248)
        x250=operator.add(x235, x249)
        x251=self.relu60(x250)
        return x251

m = M().eval()
x248 = torch.randn(torch.Size([1, 440, 7, 7]))
x235 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x248, x235)
end = time.time()
print(end-start)
