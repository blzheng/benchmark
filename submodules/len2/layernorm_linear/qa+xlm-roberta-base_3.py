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
        self.layernorm4 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear12 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x111):
        x112=self.layernorm4(x111)
        x113=self.linear12(x112)
        return x113

m = M().eval()
x111 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x111)
end = time.time()
print(end-start)
