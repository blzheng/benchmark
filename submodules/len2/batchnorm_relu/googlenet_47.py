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
        self.batchnorm2d47 = BatchNorm2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x173):
        x174=self.batchnorm2d47(x173)
        x175=torch.nn.functional.relu(x174,inplace=True)
        return x175

m = M().eval()
x173 = torch.randn(torch.Size([1, 320, 7, 7]))
start = time.time()
output = m(x173)
end = time.time()
print(end-start)
