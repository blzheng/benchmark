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
        self.sigmoid41 = Sigmoid()

    def forward(self, x648):
        x649=self.sigmoid41(x648)
        return x649

m = M().eval()
x648 = torch.randn(torch.Size([1, 2064, 1, 1]))
start = time.time()
output = m(x648)
end = time.time()
print(end-start)
