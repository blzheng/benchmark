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
        self.relu11 = ReLU(inplace=True)

    def forward(self, x26):
        x27=self.relu11(x26)
        return x27

m = M().eval()
x26 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
