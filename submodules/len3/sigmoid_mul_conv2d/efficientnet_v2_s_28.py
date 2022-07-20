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
        self.sigmoid28 = Sigmoid()
        self.conv2d163 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x517, x513):
        x518=self.sigmoid28(x517)
        x519=operator.mul(x518, x513)
        x520=self.conv2d163(x519)
        return x520

m = M().eval()
x517 = torch.randn(torch.Size([1, 1536, 1, 1]))
x513 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x517, x513)
end = time.time()
print(end-start)
