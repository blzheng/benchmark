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
        self.linear0 = Linear(in_features=25088, out_features=4096, bias=True)
        self.relu10 = ReLU(inplace=True)
        self.dropout0 = Dropout(p=0.5, inplace=False)

    def forward(self, x37):
        x38=self.linear0(x37)
        x39=self.relu10(x38)
        x40=self.dropout0(x39)
        return x40

m = M().eval()
x37 = torch.randn(torch.Size([1, 25088]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
