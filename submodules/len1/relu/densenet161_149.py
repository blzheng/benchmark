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
        self.relu149 = ReLU(inplace=True)

    def forward(self, x528):
        x529=self.relu149(x528)
        return x529

m = M().eval()
x528 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x528)
end = time.time()
print(end-start)
