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
        self.batchnorm2d40 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu627 = ReLU6(inplace=True)

    def forward(self, x116):
        x117=self.batchnorm2d40(x116)
        x118=self.relu627(x117)
        return x118

m = M().eval()
x116 = torch.randn(torch.Size([1, 576, 7, 7]))
start = time.time()
output = m(x116)
end = time.time()
print(end-start)
