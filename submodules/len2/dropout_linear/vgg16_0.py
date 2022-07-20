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
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)

    def forward(self, x35):
        x36=self.dropout0(x35)
        x37=self.linear1(x36)
        return x37

m = M().eval()
x35 = torch.randn(torch.Size([1, 4096]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)
