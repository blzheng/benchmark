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
        self.batchnorm2d4 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x27):
        x28=self.batchnorm2d4(x27)
        x29=torch.nn.functional.relu(x28,inplace=True)
        return x29

m = M().eval()
x27 = torch.randn(torch.Size([1, 192, 52, 52]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
