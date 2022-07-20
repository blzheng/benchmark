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
        self.relu8 = ReLU(inplace=True)

    def forward(self, x23, x37):
        x38=operator.add(x23, x37)
        x39=self.relu8(x38)
        return x39

m = M().eval()
x23 = torch.randn(torch.Size([1, 104, 28, 28]))
x37 = torch.randn(torch.Size([1, 104, 28, 28]))
start = time.time()
output = m(x23, x37)
end = time.time()
print(end-start)
