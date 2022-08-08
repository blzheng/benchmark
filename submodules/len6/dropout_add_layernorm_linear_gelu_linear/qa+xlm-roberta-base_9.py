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
        self.dropout26 = Dropout(p=0.1, inplace=False)
        self.layernorm17 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear52 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear53 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x397, x364):
        x398=self.dropout26(x397)
        x399=operator.add(x398, x364)
        x400=self.layernorm17(x399)
        x401=self.linear52(x400)
        x402=torch._C._nn.gelu(x401)
        x403=self.linear53(x402)
        return x403

m = M().eval()
x397 = torch.randn(torch.Size([1, 384, 768]))
x364 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x397, x364)
end = time.time()
print(end-start)
