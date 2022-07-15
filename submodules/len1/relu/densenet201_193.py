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
        self.relu193 = ReLU(inplace=True)

    def forward(self, x682):
        x683=self.relu193(x682)
        return x683

m = M().eval()
x682 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x682)
end = time.time()
print(end-start)
