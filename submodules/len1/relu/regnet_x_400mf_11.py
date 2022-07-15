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
        self.relu11 = ReLU(inplace=True)

    def forward(self, x44):
        x45=self.relu11(x44)
        return x45

m = M().eval()
x44 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
