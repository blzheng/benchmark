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
        self.conv2d78 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d46 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x235):
        x236=self.conv2d78(x235)
        x237=self.batchnorm2d46(x236)
        return x237

m = M().eval()
x235 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x235)
end = time.time()
print(end-start)
