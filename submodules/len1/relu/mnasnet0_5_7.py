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
        self.relu7 = ReLU(inplace=True)

    def forward(self, x30):
        x31=self.relu7(x30)
        return x31

m = M().eval()
x30 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x30)
end = time.time()
print(end-start)
