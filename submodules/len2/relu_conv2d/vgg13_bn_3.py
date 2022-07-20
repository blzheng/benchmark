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
        self.relu6 = ReLU(inplace=True)
        self.conv2d7 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x23):
        x24=self.relu6(x23)
        x25=self.conv2d7(x24)
        return x25

m = M().eval()
x23 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x23)
end = time.time()
print(end-start)
