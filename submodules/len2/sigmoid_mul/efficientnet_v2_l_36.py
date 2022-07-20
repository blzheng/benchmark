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
        self.sigmoid36 = Sigmoid()

    def forward(self, x697, x693):
        x698=self.sigmoid36(x697)
        x699=operator.mul(x698, x693)
        return x699

m = M().eval()
x697 = torch.randn(torch.Size([1, 2304, 1, 1]))
x693 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x697, x693)
end = time.time()
print(end-start)
