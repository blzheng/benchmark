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
        self.batchnorm2d61 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)

    def forward(self, x201):
        x202=self.batchnorm2d61(x201)
        x203=self.relu58(x202)
        return x203

m = M().eval()
x201 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x201)
end = time.time()
print(end-start)
