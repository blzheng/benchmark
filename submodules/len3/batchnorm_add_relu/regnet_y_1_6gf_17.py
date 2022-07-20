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
        self.batchnorm2d48 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu60 = ReLU(inplace=True)

    def forward(self, x246, x233):
        x247=self.batchnorm2d48(x246)
        x248=operator.add(x233, x247)
        x249=self.relu60(x248)
        return x249

m = M().eval()
x246 = torch.randn(torch.Size([1, 336, 14, 14]))
x233 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x246, x233)
end = time.time()
print(end-start)