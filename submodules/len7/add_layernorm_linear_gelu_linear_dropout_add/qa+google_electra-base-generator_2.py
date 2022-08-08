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
        self.layernorm5 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear17 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear18 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout9 = Dropout(p=0.1, inplace=False)

    def forward(self, x147, x113):
        x148=operator.add(x147, x113)
        x149=self.layernorm5(x148)
        x150=self.linear17(x149)
        x151=torch._C._nn.gelu(x150)
        x152=self.linear18(x151)
        x153=self.dropout9(x152)
        x154=operator.add(x153, x149)
        return x154

m = M().eval()
x147 = torch.randn(torch.Size([1, 384, 256]))
x113 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x147, x113)
end = time.time()
print(end-start)
