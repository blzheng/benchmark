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
        self.relu16 = ReLU(inplace=True)
        self.dropout0 = Dropout(p=0.5, inplace=False)

    def forward(self, x56):
        x57=self.relu16(x56)
        x58=self.dropout0(x57)
        return x58

m = M().eval()
x56 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x56)
end = time.time()
print(end-start)
