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
        self.conv2d37 = Conv2d(52, 208, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x114, x111):
        x115=self.conv2d37(x114)
        x116=self.sigmoid6(x115)
        x117=operator.mul(x116, x111)
        return x117

m = M().eval()
x114 = torch.randn(torch.Size([1, 52, 1, 1]))
x111 = torch.randn(torch.Size([1, 208, 14, 14]))
start = time.time()
output = m(x114, x111)
end = time.time()
print(end-start)
