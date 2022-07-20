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
        self.sigmoid39 = Sigmoid()

    def forward(self, x613, x609):
        x614=self.sigmoid39(x613)
        x615=operator.mul(x614, x609)
        return x615

m = M().eval()
x613 = torch.randn(torch.Size([1, 2304, 1, 1]))
x609 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x613, x609)
end = time.time()
print(end-start)
