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
        self.linear60 = Linear(in_features=1024, out_features=256, bias=True)
        self.dropout30 = Dropout(p=0.1, inplace=False)
        self.layernorm20 = LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.linear61 = Linear(in_features=256, out_features=256, bias=True)

    def forward(self, x445, x443):
        x446=self.linear60(x445)
        x447=self.dropout30(x446)
        x448=operator.add(x447, x443)
        x449=self.layernorm20(x448)
        x450=self.linear61(x449)
        return x450

m = M().eval()
x445 = torch.randn(torch.Size([1, 384, 1024]))
x443 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x445, x443)
end = time.time()
print(end-start)
