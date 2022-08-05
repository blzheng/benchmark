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
        self.layernorm12 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear10 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu4 = GELU(approximate='none')

    def forward(self, x111, x125):
        x126=operator.add(x111, x125)
        x127=self.layernorm12(x126)
        x128=self.linear10(x127)
        x129=self.gelu4(x128)
        return x129

m = M().eval()
x111 = torch.randn(torch.Size([1, 14, 14, 512]))
x125 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x111, x125)
end = time.time()
print(end-start)
