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
        self.relu119 = ReLU(inplace=True)

    def forward(self, x422):
        x423=self.relu119(x422)
        return x423

m = M().eval()
x422 = torch.randn(torch.Size([1, 1536, 14, 14]))
start = time.time()
output = m(x422)
end = time.time()
print(end-start)
