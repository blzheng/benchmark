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
        self.linear45 = Linear(in_features=3072, out_features=768, bias=True)

    def forward(self, x266):
        x267=self.linear45(x266)
        x268=torch.permute(x267, [0, 3, 1, 2])
        return x268

m = M().eval()
x266 = torch.randn(torch.Size([1, 14, 14, 3072]))
start = time.time()
output = m(x266)
end = time.time()
print(end-start)
