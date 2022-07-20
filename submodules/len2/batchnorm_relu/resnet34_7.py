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
        self.batchnorm2d14 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu13 = ReLU(inplace=True)

    def forward(self, x49):
        x50=self.batchnorm2d14(x49)
        x51=self.relu13(x50)
        return x51

m = M().eval()
x49 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
