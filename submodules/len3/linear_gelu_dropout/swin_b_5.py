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
        self.linear12 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu5 = GELU(approximate='none')
        self.dropout10 = Dropout(p=0.0, inplace=False)

    def forward(self, x150):
        x151=self.linear12(x150)
        x152=self.gelu5(x151)
        x153=self.dropout10(x152)
        return x153

m = M().eval()
x150 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x150)
end = time.time()
print(end-start)
