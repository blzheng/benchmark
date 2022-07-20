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
        self.relu104 = ReLU(inplace=True)

    def forward(self, x409, x423):
        x424=operator.add(x409, x423)
        x425=self.relu104(x424)
        return x425

m = M().eval()
x409 = torch.randn(torch.Size([1, 2904, 14, 14]))
x423 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x409, x423)
end = time.time()
print(end-start)
