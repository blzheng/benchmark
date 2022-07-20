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
        self.layernorm19 = LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        self.linear38 = Linear(in_features=768, out_features=3072, bias=True)
        self.gelu19 = GELU(approximate='none')
        self.linear39 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x230):
        x231=self.layernorm19(x230)
        x232=self.linear38(x231)
        x233=self.gelu19(x232)
        x234=self.linear39(x233)
        return x234

m = M().eval()
x230 = torch.randn(torch.Size([1, 14, 14, 768]))
start = time.time()
output = m(x230)
end = time.time()
print(end-start)
