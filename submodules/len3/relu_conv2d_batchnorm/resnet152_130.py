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
        self.relu130 = ReLU(inplace=True)
        self.conv2d135 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d135 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x445):
        x446=self.relu130(x445)
        x447=self.conv2d135(x446)
        x448=self.batchnorm2d135(x447)
        return x448

m = M().eval()
x445 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x445)
end = time.time()
print(end-start)
