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
        self.sigmoid25 = Sigmoid()

    def forward(self, x394, x390):
        x395=self.sigmoid25(x394)
        x396=operator.mul(x395, x390)
        return x396

m = M().eval()
x394 = torch.randn(torch.Size([1, 1200, 1, 1]))
x390 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x394, x390)
end = time.time()
print(end-start)
