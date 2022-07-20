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
        self.relu91 = ReLU()
        self.conv2d117 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d118 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x369, x367):
        x370=self.relu91(x369)
        x371=self.conv2d117(x370)
        x372=self.sigmoid22(x371)
        x373=operator.mul(x372, x367)
        x374=self.conv2d118(x373)
        return x374

m = M().eval()
x369 = torch.randn(torch.Size([1, 84, 1, 1]))
x367 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x369, x367)
end = time.time()
print(end-start)
