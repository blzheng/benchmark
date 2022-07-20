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
        self.batchnorm2d76 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x262):
        x263=self.batchnorm2d76(x262)
        x264=torch.nn.functional.relu(x263,inplace=True)
        return x264

m = M().eval()
x262 = torch.randn(torch.Size([1, 320, 5, 5]))
start = time.time()
output = m(x262)
end = time.time()
print(end-start)
