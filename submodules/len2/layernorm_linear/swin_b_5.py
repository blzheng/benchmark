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
        self.layernorm10 = LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        self.linear9 = Linear(in_features=1024, out_features=512, bias=False)

    def forward(self, x109):
        x110=self.layernorm10(x109)
        x111=self.linear9(x110)
        return x111

m = M().eval()
x109 = torch.randn(torch.Size([1, 14, 14, 1024]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
