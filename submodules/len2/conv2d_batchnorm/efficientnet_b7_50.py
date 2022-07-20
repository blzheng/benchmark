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
        self.conv2d86 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(80, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x269):
        x270=self.conv2d86(x269)
        x271=self.batchnorm2d50(x270)
        return x271

m = M().eval()
x269 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x269)
end = time.time()
print(end-start)
