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
        self.linear53 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout27 = Dropout(p=0.1, inplace=False)
        self.layernorm18 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)

    def forward(self, x402, x400):
        x403=self.linear53(x402)
        x404=self.dropout27(x403)
        x405=operator.add(x404, x400)
        x406=self.layernorm18(x405)
        return x406

m = M().eval()
x402 = torch.randn(torch.Size([1, 384, 3072]))
x400 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x402, x400)
end = time.time()
print(end-start)
