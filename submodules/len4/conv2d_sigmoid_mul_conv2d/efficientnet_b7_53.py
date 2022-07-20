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
        self.conv2d265 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid53 = Sigmoid()
        self.conv2d266 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x834, x831):
        x835=self.conv2d265(x834)
        x836=self.sigmoid53(x835)
        x837=operator.mul(x836, x831)
        x838=self.conv2d266(x837)
        return x838

m = M().eval()
x834 = torch.randn(torch.Size([1, 160, 1, 1]))
x831 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x834, x831)
end = time.time()
print(end-start)
