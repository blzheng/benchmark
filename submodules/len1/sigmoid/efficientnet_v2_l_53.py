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
        self.sigmoid53 = Sigmoid()

    def forward(self, x969):
        x970=self.sigmoid53(x969)
        return x970

m = M().eval()
x969 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x969)
end = time.time()
print(end-start)
