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
        self.linear3 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)

    def forward(self, x60, x28):
        x61=self.linear3(x60)
        x62=self.dropout2(x61)
        x63=operator.add(x62, x28)
        return x63

m = M().eval()
x60 = torch.randn(torch.Size([1, 384, 768]))
x28 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x60, x28)
end = time.time()
print(end-start)