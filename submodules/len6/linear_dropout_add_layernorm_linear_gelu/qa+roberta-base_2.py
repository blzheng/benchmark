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
        self.linear15 = Linear(in_features=768, out_features=768, bias=True)
        self.dropout8 = Dropout(p=0.1, inplace=False)
        self.layernorm5 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear16 = Linear(in_features=768, out_features=3072, bias=True)

    def forward(self, x144, x112):
        x145=self.linear15(x144)
        x146=self.dropout8(x145)
        x147=operator.add(x146, x112)
        x148=self.layernorm5(x147)
        x149=self.linear16(x148)
        x150=torch._C._nn.gelu(x149)
        return x150

m = M().eval()
x144 = torch.randn(torch.Size([1, 384, 768]))
x112 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x144, x112)
end = time.time()
print(end-start)
