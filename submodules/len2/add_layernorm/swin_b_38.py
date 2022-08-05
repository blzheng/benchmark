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
        self.layernorm44 = LayerNorm((512,), eps=1e-05, elementwise_affine=True)

    def forward(self, x479, x493):
        x494=operator.add(x479, x493)
        x495=self.layernorm44(x494)
        return x495

m = M().eval()
x479 = torch.randn(torch.Size([1, 14, 14, 512]))
x493 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x479, x493)
end = time.time()
print(end-start)
