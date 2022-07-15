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
        self.relu151 = ReLU(inplace=True)

    def forward(self, x535):
        x536=self.relu151(x535)
        return x536

m = M().eval()
x535 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x535)
end = time.time()
print(end-start)
