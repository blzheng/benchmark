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
        self.layernorm36 = LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        self.linear34 = Linear(in_features=384, out_features=1536, bias=True)
        self.gelu16 = GELU(approximate='none')

    def forward(self, x402):
        x403=self.layernorm36(x402)
        x404=self.linear34(x403)
        x405=self.gelu16(x404)
        return x405

m = M().eval()
x402 = torch.randn(torch.Size([1, 14, 14, 384]))
start = time.time()
output = m(x402)
end = time.time()
print(end-start)
