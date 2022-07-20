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
        self.conv2d35 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x369, x359):
        x370=operator.add(x369, x359)
        x372=self.conv2d35(x370)
        return x372

m = M().eval()
x369 = torch.randn(torch.Size([1, 768, 14, 14]))
x359 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x369, x359)
end = time.time()
print(end-start)
