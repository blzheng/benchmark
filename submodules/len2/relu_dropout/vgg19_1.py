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
        self.relu17 = ReLU(inplace=True)
        self.dropout1 = Dropout(p=0.5, inplace=False)

    def forward(self, x43):
        x44=self.relu17(x43)
        x45=self.dropout1(x44)
        return x45

m = M().eval()
x43 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x43)
end = time.time()
print(end-start)
