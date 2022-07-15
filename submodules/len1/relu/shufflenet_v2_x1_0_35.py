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
        self.relu35 = ReLU(inplace=True)

    def forward(self, x350):
        x351=self.relu35(x350)
        return x351

m = M().eval()
x350 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x350)
end = time.time()
print(end-start)
