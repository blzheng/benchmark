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
        self.linear65 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear66 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout33 = Dropout(p=0.1, inplace=False)
        self.layernorm22 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear67 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x485, x485):
        x486=self.linear65(x485)
        x487=torch._C._nn.gelu(x486)
        x488=self.linear66(x487)
        x489=self.dropout33(x488)
        x490=operator.add(x489, x485)
        x491=self.layernorm22(x490)
        x492=self.linear67(x491)
        return x492

m = M().eval()
x485 = torch.randn(torch.Size([1, 384, 256]))
x485 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x485, x485)
end = time.time()
print(end-start)
