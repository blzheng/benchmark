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
        self.relu64 = ReLU(inplace=True)

    def forward(self, x221):
        x222=self.relu64(x221)
        return x222

m = M().eval()
x221 = torch.randn(torch.Size([1, 720, 14, 14]))
start = time.time()
output = m(x221)
end = time.time()
print(end-start)