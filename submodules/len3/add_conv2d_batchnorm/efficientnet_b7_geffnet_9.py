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
        self.conv2d62 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d36 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x186, x172):
        x187=operator.add(x186, x172)
        x188=self.conv2d62(x187)
        x189=self.batchnorm2d36(x188)
        return x189

m = M().eval()
x186 = torch.randn(torch.Size([1, 80, 28, 28]))
x172 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x186, x172)
end = time.time()
print(end-start)
