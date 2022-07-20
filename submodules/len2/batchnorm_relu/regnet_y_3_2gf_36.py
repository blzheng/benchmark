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
        self.batchnorm2d56 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu70 = ReLU(inplace=True)

    def forward(self, x285):
        x286=self.batchnorm2d56(x285)
        x287=self.relu70(x286)
        return x287

m = M().eval()
x285 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x285)
end = time.time()
print(end-start)
