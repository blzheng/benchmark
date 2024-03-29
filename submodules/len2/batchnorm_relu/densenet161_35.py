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
        self.batchnorm2d35 = BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu35 = ReLU(inplace=True)

    def forward(self, x125):
        x126=self.batchnorm2d35(x125)
        x127=self.relu35(x126)
        return x127

m = M().eval()
x125 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x125)
end = time.time()
print(end-start)
