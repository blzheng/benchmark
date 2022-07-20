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
        self.batchnorm2d9 = BatchNorm2d(96, eps=1e-05, momentum=0.00029999999999996696, affine=True, track_running_stats=True)
        self.relu6 = ReLU(inplace=True)

    def forward(self, x26):
        x27=self.batchnorm2d9(x26)
        x28=self.relu6(x27)
        return x28

m = M().eval()
x26 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
