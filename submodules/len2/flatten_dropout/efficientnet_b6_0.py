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
        self.dropout0 = Dropout(p=0.5, inplace=True)

    def forward(self, x704):
        x705=torch.flatten(x704, 1)
        x706=self.dropout0(x705)
        return x706

m = M().eval()
x704 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x704)
end = time.time()
print(end-start)
