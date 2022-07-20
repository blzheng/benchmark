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
        self.conv2d88 = Conv2d(480, 480, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=480, bias=False)
        self.batchnorm2d52 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x265):
        x266=self.conv2d88(x265)
        x267=self.batchnorm2d52(x266)
        return x267

m = M().eval()
x265 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x265)
end = time.time()
print(end-start)
