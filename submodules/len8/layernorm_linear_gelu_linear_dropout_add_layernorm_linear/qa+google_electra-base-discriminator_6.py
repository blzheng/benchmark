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
        self.layernorm13 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear40 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear41 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout21 = Dropout(p=0.1, inplace=False)
        self.layernorm14 = LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        self.linear42 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x315):
        x316=self.layernorm13(x315)
        x317=self.linear40(x316)
        x318=torch._C._nn.gelu(x317)
        x319=self.linear41(x318)
        x320=self.dropout21(x319)
        x321=operator.add(x320, x316)
        x322=self.layernorm14(x321)
        x323=self.linear42(x322)
        return x323

m = M().eval()
x315 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x315)
end = time.time()
print(end-start)
