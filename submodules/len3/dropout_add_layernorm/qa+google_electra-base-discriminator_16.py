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
        self.dropout24 = Dropout(p=0.1, inplace=False)
        self.layernorm16 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x361, x358):
        x362=self.dropout24(x361)
        x363=operator.add(x362, x358)
        x364=self.layernorm16(x363)
        return x364

m = M().eval()
x361 = torch.randn(torch.Size([1, 384, 768]))
x358 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x361, x358)
end = time.time()
print(end-start)
