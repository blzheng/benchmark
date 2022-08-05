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
        self.relu19 = ReLU(inplace=True)

    def forward(self, x78, x70):
        x79=operator.add(x78, x70)
        x80=self.relu19(x79)
        return x80

m = M().eval()
x78 = torch.randn(torch.Size([1, 512, 28, 28]))
x70 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x78, x70)
end = time.time()
print(end-start)
