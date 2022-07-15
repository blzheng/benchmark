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
        self.relu122 = ReLU(inplace=True)

    def forward(self, x434):
        x435=self.relu122(x434)
        return x435

m = M().eval()
x434 = torch.randn(torch.Size([1, 1296, 7, 7]))
start = time.time()
output = m(x434)
end = time.time()
print(end-start)
