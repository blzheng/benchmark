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
        self.batchnorm2d70 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)

    def forward(self, x233):
        x234=self.batchnorm2d70(x233)
        x235=self.relu67(x234)
        return x235

m = M().eval()
x233 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x233)
end = time.time()
print(end-start)
