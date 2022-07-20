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
        self.sigmoid35 = Sigmoid()

    def forward(self, x681):
        x682=self.sigmoid35(x681)
        return x682

m = M().eval()
x681 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x681)
end = time.time()
print(end-start)