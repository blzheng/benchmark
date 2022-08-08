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
        self.layernorm0 = LayerNorm((128,), eps=1e-12, elementwise_affine=True)
        self.dropout0 = Dropout(p=0.1, inplace=False)

    def forward(self, x26):
        x27=self.layernorm0(x26)
        x28=self.dropout0(x27)
        return x28

m = M().eval()
x26 = torch.randn(torch.Size([1, 384, 128]))
start = time.time()
output = m(x26)
end = time.time()
print(end-start)
