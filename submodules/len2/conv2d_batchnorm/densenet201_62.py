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
        self.conv2d125 = Conv2d(1632, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d126 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x444):
        x445=self.conv2d125(x444)
        x446=self.batchnorm2d126(x445)
        return x446

m = M().eval()
x444 = torch.randn(torch.Size([1, 1632, 14, 14]))
start = time.time()
output = m(x444)
end = time.time()
print(end-start)
