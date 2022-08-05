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
        self.linear9 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout5 = Dropout(p=0.1, inplace=False)

    def forward(self, x98, x101):
        x102=x98.view(x101)
        x103=self.linear9(x102)
        x104=self.dropout5(x103)
        return x104

m = M().eval()
x98 = torch.randn(torch.Size([1, 384, 12, 64]))
x101 = (1, 384, 768, )
start = time.time()
output = m(x98, x101)
end = time.time()
print(end-start)
