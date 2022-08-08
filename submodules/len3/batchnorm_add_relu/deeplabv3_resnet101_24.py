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
        self.batchnorm2d69 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)

    def forward(self, x229, x222):
        x230=self.batchnorm2d69(x229)
        x231=operator.add(x230, x222)
        x232=self.relu64(x231)
        return x232

m = M().eval()
x229 = torch.randn(torch.Size([1, 1024, 28, 28]))
x222 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x229, x222)
end = time.time()
print(end-start)
