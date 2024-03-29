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
        self.sigmoid47 = Sigmoid()

    def forward(self, x741, x737):
        x742=self.sigmoid47(x741)
        x743=operator.mul(x742, x737)
        return x743

m = M().eval()
x741 = torch.randn(torch.Size([1, 2304, 1, 1]))
x737 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x741, x737)
end = time.time()
print(end-start)
