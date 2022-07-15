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
        self.relu136 = ReLU(inplace=True)

    def forward(self, x483):
        x484=self.relu136(x483)
        return x484

m = M().eval()
x483 = torch.randn(torch.Size([1, 896, 7, 7]))
start = time.time()
output = m(x483)
end = time.time()
print(end-start)
