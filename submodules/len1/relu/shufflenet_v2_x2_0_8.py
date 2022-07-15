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

    def forward(self, x77):
        x78=self.relu8(x77)
        return x78

m = M().eval()
x77 = torch.randn(torch.Size([1, 122, 28, 28]))
start = time.time()
output = m(x77)
end = time.time()
print(end-start)
