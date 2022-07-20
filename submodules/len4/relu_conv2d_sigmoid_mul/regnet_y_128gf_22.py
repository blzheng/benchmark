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
        self.conv2d117 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()

    def forward(self, x369, x367):
        x370=self.relu91(x369)
        x371=self.conv2d117(x370)
        x372=self.sigmoid22(x371)
        x373=operator.mul(x372, x367)
        return x373

m = M().eval()
x369 = torch.randn(torch.Size([1, 726, 1, 1]))
x367 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x369, x367)
end = time.time()
print(end-start)