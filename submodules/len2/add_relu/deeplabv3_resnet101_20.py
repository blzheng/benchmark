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
        self.relu61 = ReLU(inplace=True)

    def forward(self, x220, x212):
        x221=operator.add(x220, x212)
        x222=self.relu61(x221)
        return x222

m = M().eval()
x220 = torch.randn(torch.Size([1, 1024, 28, 28]))
x212 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x220, x212)
end = time.time()
print(end-start)
