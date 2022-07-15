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

    def forward(self, x276):
        x277=self.relu27(x276)
        return x277

m = M().eval()
x276 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x276)
end = time.time()
print(end-start)
