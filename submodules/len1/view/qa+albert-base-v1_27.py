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

    def forward(self, x363, x368):
        x369=x363.view(x368)
        return x369

m = M().eval()
x363 = torch.randn(torch.Size([1, 384, 768]))
x368 = (1, 384, 12, 64, )
start = time.time()
output = m(x363, x368)
end = time.time()
print(end-start)
