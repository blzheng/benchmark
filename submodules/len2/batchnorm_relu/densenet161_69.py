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
        self.batchnorm2d69 = BatchNorm2d(1104, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu69 = ReLU(inplace=True)

    def forward(self, x246):
        x247=self.batchnorm2d69(x246)
        x248=self.relu69(x247)
        return x248

m = M().eval()
x246 = torch.randn(torch.Size([1, 1104, 14, 14]))
start = time.time()
output = m(x246)
end = time.time()
print(end-start)
