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
        self.layernorm36 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.linear34 = Linear(in_features=512, out_features=2048, bias=True)
        self.gelu16 = GELU(approximate='none')
        self.dropout32 = Dropout(p=0.0, inplace=False)
        self.linear35 = Linear(in_features=2048, out_features=512, bias=True)

    def forward(self, x387, x401):
        x402=operator.add(x387, x401)
        x403=self.layernorm36(x402)
        x404=self.linear34(x403)
        x405=self.gelu16(x404)
        x406=self.dropout32(x405)
        x407=self.linear35(x406)
        return x407

m = M().eval()
x387 = torch.randn(torch.Size([1, 14, 14, 512]))
x401 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x387, x401)
end = time.time()
print(end-start)
