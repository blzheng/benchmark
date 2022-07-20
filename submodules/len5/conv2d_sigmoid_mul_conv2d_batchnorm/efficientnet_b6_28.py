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
        self.conv2d141 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid28 = Sigmoid()
        self.conv2d142 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d84 = BatchNorm2d(200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x441, x438):
        x442=self.conv2d141(x441)
        x443=self.sigmoid28(x442)
        x444=operator.mul(x443, x438)
        x445=self.conv2d142(x444)
        x446=self.batchnorm2d84(x445)
        return x446

m = M().eval()
x441 = torch.randn(torch.Size([1, 50, 1, 1]))
x438 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x441, x438)
end = time.time()
print(end-start)
