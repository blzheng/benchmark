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
        self.batchnorm2d78 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu74 = ReLU(inplace=True)

    def forward(self, x255):
        x256=self.batchnorm2d78(x255)
        x257=self.relu74(x256)
        return x257

m = M().eval()
x255 = torch.randn(torch.Size([1, 1008, 7, 7]))
start = time.time()
output = m(x255)
end = time.time()
print(end-start)