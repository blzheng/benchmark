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
        self.linear17 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout9 = Dropout(p=0.1, inplace=False)

    def forward(self, x150, x148):
        x151=self.linear17(x150)
        x152=self.dropout9(x151)
        x153=operator.add(x152, x148)
        return x153

m = M().eval()
x150 = torch.randn(torch.Size([1, 384, 3072]))
x148 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x150, x148)
end = time.time()
print(end-start)
