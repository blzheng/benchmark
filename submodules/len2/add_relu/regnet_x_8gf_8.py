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

    def forward(self, x89, x97):
        x98=operator.add(x89, x97)
        x99=self.relu27(x98)
        return x99

m = M().eval()
x89 = torch.randn(torch.Size([1, 720, 14, 14]))
x97 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x89, x97)
end = time.time()
print(end-start)
