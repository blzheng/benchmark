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
        self.batchnorm2d2 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x20):
        x21=self.batchnorm2d2(x20)
        x22=torch.nn.functional.relu(x21,inplace=True)
        return x22

m = M().eval()
x20 = torch.randn(torch.Size([1, 64, 109, 109]))
start = time.time()
output = m(x20)
end = time.time()
print(end-start)
