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
        self.dropout14 = Dropout(p=0.1, inplace=False)
        self.layernorm9 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear28 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x229, x196):
        x230=self.dropout14(x229)
        x231=operator.add(x230, x196)
        x232=self.layernorm9(x231)
        x233=self.linear28(x232)
        x234=torch._C._nn.gelu(x233)
        return x234

m = M().eval()
x229 = torch.randn(torch.Size([1, 384, 768]))
x196 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x229, x196)
end = time.time()
print(end-start)
