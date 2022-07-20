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
        self.conv2d51 = Conv2d(208, 52, kernel_size=(1, 1), stride=(1, 1))
        self.relu39 = ReLU()
        self.conv2d52 = Conv2d(52, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x160, x159):
        x161=self.conv2d51(x160)
        x162=self.relu39(x161)
        x163=self.conv2d52(x162)
        x164=self.sigmoid9(x163)
        x165=operator.mul(x164, x159)
        return x165

m = M().eval()
x160 = torch.randn(torch.Size([1, 208, 1, 1]))
x159 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x160, x159)
end = time.time()
print(end-start)
