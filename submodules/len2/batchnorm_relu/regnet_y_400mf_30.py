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
        self.batchnorm2d48 = BatchNorm2d(440, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu58 = ReLU(inplace=True)

    def forward(self, x239):
        x240=self.batchnorm2d48(x239)
        x241=self.relu58(x240)
        return x241

m = M().eval()
x239 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x239)
end = time.time()
print(end-start)
