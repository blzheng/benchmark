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
        self.linear6 = Linear(in_features=3072, out_features=768, bias=True)
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x395, x394):
        x396=torch._C._nn.gelu(x395)
        x397=self.linear6(x396)
        x398=operator.add(x397, x394)
        x399=self.layernorm2(x398)
        x400=self.linear1(x399)
        return x400

m = M().eval()
x395 = torch.randn(torch.Size([1, 384, 3072]))
x394 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x395, x394)
end = time.time()
print(end-start)
