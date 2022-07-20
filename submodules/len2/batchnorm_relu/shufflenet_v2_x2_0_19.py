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
        self.batchnorm2d29 = BatchNorm2d(244, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu19 = ReLU(inplace=True)

    def forward(self, x188):
        x189=self.batchnorm2d29(x188)
        x190=self.relu19(x189)
        return x190

m = M().eval()
x188 = torch.randn(torch.Size([1, 244, 14, 14]))
start = time.time()
output = m(x188)
end = time.time()
print(end-start)
