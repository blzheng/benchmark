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
        self.batchnorm2d52 = BatchNorm2d(480, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x276):
        x277=self.conv2d88(x276)
        x278=self.batchnorm2d52(x277)
        return x278

m = M().eval()
x276 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
