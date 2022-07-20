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
        self.sigmoid34 = Sigmoid()

    def forward(self, x536, x532):
        x537=self.sigmoid34(x536)
        x538=operator.mul(x537, x532)
        return x538

m = M().eval()
x536 = torch.randn(torch.Size([1, 2064, 1, 1]))
x532 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x536, x532)
end = time.time()
print(end-start)
