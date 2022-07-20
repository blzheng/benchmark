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
        self.relu125 = ReLU(inplace=True)
        self.conv2d125 = Conv2d(1632, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d126 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu126 = ReLU(inplace=True)

    def forward(self, x443):
        x444=self.relu125(x443)
        x445=self.conv2d125(x444)
        x446=self.batchnorm2d126(x445)
        x447=self.relu126(x446)
        return x447

m = M().eval()
x443 = torch.randn(torch.Size([1, 1632, 14, 14]))
start = time.time()
output = m(x443)
end = time.time()
print(end-start)
