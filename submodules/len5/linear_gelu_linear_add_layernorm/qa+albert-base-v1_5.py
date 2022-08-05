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
        self.linear5 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x246, x246):
        x247=self.linear5(x246)
        x248=torch._C._nn.gelu(x247)
        x249=self.linear6(x248)
        x250=operator.add(x249, x246)
        x251=self.layernorm2(x250)
        return x251

m = M().eval()
x246 = torch.randn(torch.Size([1, 384, 768]))
x246 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x246, x246)
end = time.time()
print(end-start)
