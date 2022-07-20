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
        self.conv2d107 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()

    def forward(self, x345, x342):
        x346=self.conv2d107(x345)
        x347=self.sigmoid16(x346)
        x348=operator.mul(x347, x342)
        return x348

m = M().eval()
x345 = torch.randn(torch.Size([1, 44, 1, 1]))
x342 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x345, x342)
end = time.time()
print(end-start)
