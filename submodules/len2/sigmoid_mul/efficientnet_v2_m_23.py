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
        self.sigmoid23 = Sigmoid()

    def forward(self, x456, x452):
        x457=self.sigmoid23(x456)
        x458=operator.mul(x457, x452)
        return x458

m = M().eval()
x456 = torch.randn(torch.Size([1, 1824, 1, 1]))
x452 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x456, x452)
end = time.time()
print(end-start)
