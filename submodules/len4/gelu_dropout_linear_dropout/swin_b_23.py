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
        self.gelu23 = GELU(approximate='none')
        self.dropout46 = Dropout(p=0.0, inplace=False)
        self.linear50 = Linear(in_features=4096, out_features=1024, bias=True)
        self.dropout47 = Dropout(p=0.0, inplace=False)

    def forward(self, x573):
        x574=self.gelu23(x573)
        x575=self.dropout46(x574)
        x576=self.linear50(x575)
        x577=self.dropout47(x576)
        return x577

m = M().eval()
x573 = torch.randn(torch.Size([1, 7, 7, 4096]))
start = time.time()
output = m(x573)
end = time.time()
print(end-start)
