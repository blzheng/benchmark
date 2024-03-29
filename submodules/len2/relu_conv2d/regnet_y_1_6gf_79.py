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
        self.relu105 = ReLU(inplace=True)
        self.conv2d136 = Conv2d(888, 888, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=37, bias=False)

    def forward(self, x429):
        x430=self.relu105(x429)
        x431=self.conv2d136(x430)
        return x431

m = M().eval()
x429 = torch.randn(torch.Size([1, 888, 7, 7]))
start = time.time()
output = m(x429)
end = time.time()
print(end-start)
