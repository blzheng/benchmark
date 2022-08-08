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
        self.linear4 = Linear(in_features=256, out_features=256, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=256, out_features=1024, bias=True)

    def forward(self, x57, x60, x29):
        x61=x57.view(x60)
        x62=self.linear4(x61)
        x63=self.dropout2(x62)
        x64=operator.add(x63, x29)
        x65=self.layernorm1(x64)
        x66=self.linear5(x65)
        return x66

m = M().eval()
x57 = torch.randn(torch.Size([1, 384, 4, 64]))
x60 = (1, 384, 256, )
x29 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x57, x60, x29)
end = time.time()
print(end-start)
