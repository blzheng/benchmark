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
        self.linear53 = Linear(in_features=256, out_features=1024, bias=True)
        self.linear54 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout27 = Dropout(p=0.1, inplace=False)
        self.layernorm18 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear55 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x401, x401):
        x402=self.linear53(x401)
        x403=torch._C._nn.gelu(x402)
        x404=self.linear54(x403)
        x405=self.dropout27(x404)
        x406=operator.add(x405, x401)
        x407=self.layernorm18(x406)
        x408=self.linear55(x407)
        return x408

m = M().eval()
x401 = torch.randn(torch.Size([1, 384, 256]))
x401 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x401, x401)
end = time.time()
print(end-start)
