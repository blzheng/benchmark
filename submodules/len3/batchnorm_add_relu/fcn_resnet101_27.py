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
        self.batchnorm2d78 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu73 = ReLU(inplace=True)

    def forward(self, x259, x252):
        x260=self.batchnorm2d78(x259)
        x261=operator.add(x260, x252)
        x262=self.relu73(x261)
        return x262

m = M().eval()
x259 = torch.randn(torch.Size([1, 1024, 28, 28]))
x252 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x259, x252)
end = time.time()
print(end-start)
