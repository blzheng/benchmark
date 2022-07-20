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
        self.batchnorm2d110 = BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu110 = ReLU(inplace=True)

    def forward(self, x391):
        x392=self.batchnorm2d110(x391)
        x393=self.relu110(x392)
        return x393

m = M().eval()
x391 = torch.randn(torch.Size([1, 864, 7, 7]))
start = time.time()
output = m(x391)
end = time.time()
print(end-start)
