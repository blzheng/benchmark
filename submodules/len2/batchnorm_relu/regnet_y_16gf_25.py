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
        self.batchnorm2d40 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)

    def forward(self, x202):
        x203=self.batchnorm2d40(x202)
        x204=self.relu49(x203)
        return x204

m = M().eval()
x202 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x202)
end = time.time()
print(end-start)
