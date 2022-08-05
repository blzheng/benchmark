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

    def forward(self, x174, x172):
        x175=self.linear6(x174)
        x176=operator.add(x175, x172)
        x177=self.layernorm2(x176)
        x178=self.linear1(x177)
        return x178

m = M().eval()
x174 = torch.randn(torch.Size([1, 384, 3072]))
x172 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x174, x172)
end = time.time()
print(end-start)
