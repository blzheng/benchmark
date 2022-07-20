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
        self.linear0 = Linear(in_features=2304, out_features=1000, bias=True)

    def forward(self, x704):
        x705=torch.flatten(x704, 1)
        x706=self.dropout0(x705)
        x707=self.linear0(x706)
        return x707

m = M().eval()
x704 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x704)
end = time.time()
print(end-start)
