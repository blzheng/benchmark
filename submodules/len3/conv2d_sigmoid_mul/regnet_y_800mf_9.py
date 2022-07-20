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
        self.conv2d52 = Conv2d(80, 320, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x162, x159):
        x163=self.conv2d52(x162)
        x164=self.sigmoid9(x163)
        x165=operator.mul(x164, x159)
        return x165

m = M().eval()
x162 = torch.randn(torch.Size([1, 80, 1, 1]))
x159 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x162, x159)
end = time.time()
print(end-start)
