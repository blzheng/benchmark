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
        self.layernorm17 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear53 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear54 = Linear(in_features=1024, out_features=256, bias=True)

    def forward(self, x400):
        x401=self.layernorm17(x400)
        x402=self.linear53(x401)
        x403=torch._C._nn.gelu(x402)
        x404=self.linear54(x403)
        return x404

m = M().eval()
x400 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x400)
end = time.time()
print(end-start)
