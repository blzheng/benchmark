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
        self.conv2d166 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d98 = BatchNorm2d(224, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x491, x496):
        x497=operator.mul(x491, x496)
        x498=self.conv2d166(x497)
        x499=self.batchnorm2d98(x498)
        return x499

m = M().eval()
x491 = torch.randn(torch.Size([1, 1344, 14, 14]))
x496 = torch.randn(torch.Size([1, 1344, 1, 1]))
start = time.time()
output = m(x491, x496)
end = time.time()
print(end-start)
