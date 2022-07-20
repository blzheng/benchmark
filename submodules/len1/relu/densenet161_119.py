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

    def forward(self, x423):
        x424=self.relu119(x423)
        return x424

m = M().eval()
x423 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x423)
end = time.time()
print(end-start)