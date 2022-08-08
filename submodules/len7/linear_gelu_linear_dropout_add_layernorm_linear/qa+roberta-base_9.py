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
        self.linear58 = Linear(in_features=768, out_features=3072, bias=True)
        self.linear59 = Linear(in_features=3072, out_features=768, bias=True)
        self.dropout30 = Dropout(p=0.1, inplace=False)
        self.layernorm20 = LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.linear60 = Linear(in_features=768, out_features=768, bias=True)

    def forward(self, x442, x442):
        x443=self.linear58(x442)
        x444=torch._C._nn.gelu(x443)
        x445=self.linear59(x444)
        x446=self.dropout30(x445)
        x447=operator.add(x446, x442)
        x448=self.layernorm20(x447)
        x449=self.linear60(x448)
        return x449

m = M().eval()
x442 = torch.randn(torch.Size([1, 384, 768]))
x442 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x442, x442)
end = time.time()
print(end-start)
