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
        self.layernorm2 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear1 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x435):
        x436=self.layernorm2(x435)
        x437=self.linear1(x436)
        return x437

m = M().eval()
x435 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x435)
end = time.time()
print(end-start)
