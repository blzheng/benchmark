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
        self.relu21 = ReLU(inplace=True)

    def forward(self, x69, x77):
        x78=operator.add(x69, x77)
        x79=self.relu21(x78)
        return x79

m = M().eval()
x69 = torch.randn(torch.Size([1, 160, 14, 14]))
x77 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x69, x77)
end = time.time()
print(end-start)
