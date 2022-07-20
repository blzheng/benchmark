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
        self.conv2d60 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x180, x189, x204, x208):
        x209=torch.cat([x180, x189, x204, x208], 1)
        x210=self.conv2d60(x209)
        return x210

m = M().eval()
x180 = torch.randn(torch.Size([1, 192, 12, 12]))
x189 = torch.randn(torch.Size([1, 192, 12, 12]))
x204 = torch.randn(torch.Size([1, 192, 12, 12]))
x208 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x180, x189, x204, x208)
end = time.time()
print(end-start)
