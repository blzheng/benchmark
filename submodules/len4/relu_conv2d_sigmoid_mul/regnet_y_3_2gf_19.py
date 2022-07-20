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
        self.relu79 = ReLU()
        self.conv2d102 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()

    def forward(self, x321, x319):
        x322=self.relu79(x321)
        x323=self.conv2d102(x322)
        x324=self.sigmoid19(x323)
        x325=operator.mul(x324, x319)
        return x325

m = M().eval()
x321 = torch.randn(torch.Size([1, 144, 1, 1]))
x319 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x321, x319)
end = time.time()
print(end-start)
