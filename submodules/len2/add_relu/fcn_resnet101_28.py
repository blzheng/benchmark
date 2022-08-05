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
        self.relu85 = ReLU(inplace=True)

    def forward(self, x300, x292):
        x301=operator.add(x300, x292)
        x302=self.relu85(x301)
        return x302

m = M().eval()
x300 = torch.randn(torch.Size([1, 1024, 28, 28]))
x292 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x300, x292)
end = time.time()
print(end-start)
