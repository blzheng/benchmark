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
        self.layernorm6 = LayerNorm((512,), eps=1e-06, elementwise_affine=True)
        self.linear12 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu6 = GELU(approximate='none')

    def forward(self, x87):
        x88=self.layernorm6(x87)
        x89=self.linear12(x88)
        x90=self.gelu6(x89)
        return x90

m = M().eval()
x87 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
