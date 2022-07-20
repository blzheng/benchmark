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
        self.relu27 = ReLU(inplace=True)

    def forward(self, x87, x95):
        x96=operator.add(x87, x95)
        x97=self.relu27(x96)
        return x97

m = M().eval()
x87 = torch.randn(torch.Size([1, 672, 28, 28]))
x95 = torch.randn(torch.Size([1, 672, 28, 28]))
start = time.time()
output = m(x87, x95)
end = time.time()
print(end-start)
