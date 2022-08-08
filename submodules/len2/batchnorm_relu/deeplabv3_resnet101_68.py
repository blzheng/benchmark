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
        self.batchnorm2d105 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu101 = ReLU()

    def forward(self, x348):
        x349=self.batchnorm2d105(x348)
        x350=self.relu101(x349)
        return x350

m = M().eval()
x348 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x348)
end = time.time()
print(end-start)