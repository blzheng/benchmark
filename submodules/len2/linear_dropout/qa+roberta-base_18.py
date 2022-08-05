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
        self.linear57 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout29 = Dropout(p=0.1, inplace=False)

    def forward(self, x438):
        x439=self.linear57(x438)
        x440=self.dropout29(x439)
        return x440

m = M().eval()
x438 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x438)
end = time.time()
print(end-start)
