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
        self.batchnorm2d27 = BatchNorm2d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu27 = ReLU(inplace=True)

    def forward(self, x88, x97):
        x89=self.batchnorm2d27(x88)
        x98=operator.add(x89, x97)
        x99=self.relu27(x98)
        return x99

m = M().eval()
x88 = torch.randn(torch.Size([1, 432, 14, 14]))
x97 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x88, x97)
end = time.time()
print(end-start)
