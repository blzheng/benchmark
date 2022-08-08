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
        self.linear4 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout2 = Dropout(p=0.1, inplace=False)
        self.layernorm1 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x352, x325):
        x353=x352.flatten(2)
        x354=self.linear4(x353)
        x355=self.dropout2(x354)
        x356=operator.add(x325, x355)
        x357=self.layernorm1(x356)
        x358=self.linear5(x357)
        x359=torch._C._nn.gelu(x358)
        return x359

m = M().eval()
x352 = torch.randn(torch.Size([1, 384, 12, 64]))
x325 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x352, x325)
end = time.time()
print(end-start)
