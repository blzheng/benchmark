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
        self.batchnorm2d68 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu64 = ReLU(inplace=True)

    def forward(self, x224):
        x225=self.batchnorm2d68(x224)
        x226=self.relu64(x225)
        return x226

m = M().eval()
x224 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
