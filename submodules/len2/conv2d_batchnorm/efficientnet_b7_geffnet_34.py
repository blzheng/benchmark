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
        self.conv2d58 = Conv2d(480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
        self.batchnorm2d34 = BatchNorm2d(480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x175):
        x176=self.conv2d58(x175)
        x177=self.batchnorm2d34(x176)
        return x177

m = M().eval()
x175 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
