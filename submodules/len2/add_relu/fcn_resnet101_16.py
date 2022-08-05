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
        self.relu49 = ReLU(inplace=True)

    def forward(self, x180, x172):
        x181=operator.add(x180, x172)
        x182=self.relu49(x181)
        return x182

m = M().eval()
x180 = torch.randn(torch.Size([1, 1024, 28, 28]))
x172 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x180, x172)
end = time.time()
print(end-start)
