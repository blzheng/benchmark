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
        self.relu13 = ReLU(inplace=True)
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.linear1 = Linear(in_features=4096, out_features=4096, bias=True)

    def forward(self, x33):
        x34=self.linear0(x33)
        x35=self.relu13(x34)
        x36=self.dropout0(x35)
        x37=self.linear1(x36)
        return x37

m = M().eval()
x33 = torch.randn(torch.Size([1, 25088]))
start = time.time()
output = m(x33)
end = time.time()
print(end-start)
