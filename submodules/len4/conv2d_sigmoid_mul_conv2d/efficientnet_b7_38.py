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
        self.conv2d190 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid38 = Sigmoid()
        self.conv2d191 = Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x598, x595):
        x599=self.conv2d190(x598)
        x600=self.sigmoid38(x599)
        x601=operator.mul(x600, x595)
        x602=self.conv2d191(x601)
        return x602

m = M().eval()
x598 = torch.randn(torch.Size([1, 56, 1, 1]))
x595 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x598, x595)
end = time.time()
print(end-start)
