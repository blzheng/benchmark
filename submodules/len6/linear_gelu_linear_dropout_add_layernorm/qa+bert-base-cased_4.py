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
        self.linear28 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear29 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout15 = Dropout(p=0.1, inplace=False)
        self.layernorm10 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x232, x232):
        x233=self.linear28(x232)
        x234=torch._C._nn.gelu(x233)
        x235=self.linear29(x234)
        x236=self.dropout15(x235)
        x237=operator.add(x236, x232)
        x238=self.layernorm10(x237)
        return x238

m = M().eval()
x232 = torch.randn(torch.Size([1, 384, 768]))
x232 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x232, x232)
end = time.time()
print(end-start)
