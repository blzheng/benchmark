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
        self.batchnorm2d128 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu128 = ReLU(inplace=True)

    def forward(self, x452):
        x453=self.batchnorm2d128(x452)
        x454=self.relu128(x453)
        return x454

m = M().eval()
x452 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x452)
end = time.time()
print(end-start)
