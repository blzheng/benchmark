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
        self.conv2d33 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x97, x94):
        x98=self.conv2d33(x97)
        x99=self.sigmoid6(x98)
        x100=operator.mul(x99, x94)
        return x100

m = M().eval()
x97 = torch.randn(torch.Size([1, 20, 1, 1]))
x94 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x97, x94)
end = time.time()
print(end-start)
