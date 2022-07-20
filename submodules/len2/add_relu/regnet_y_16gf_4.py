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
        self.relu20 = ReLU(inplace=True)

    def forward(self, x71, x85):
        x86=operator.add(x71, x85)
        x87=self.relu20(x86)
        return x87

m = M().eval()
x71 = torch.randn(torch.Size([1, 448, 28, 28]))
x85 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x71, x85)
end = time.time()
print(end-start)
