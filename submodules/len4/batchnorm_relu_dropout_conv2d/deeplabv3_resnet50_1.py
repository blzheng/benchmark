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
        self.batchnorm2d60 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu56 = ReLU()
        self.dropout1 = Dropout(p=0.1, inplace=False)
        self.conv2d62 = Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x204):
        x205=self.batchnorm2d60(x204)
        x206=self.relu56(x205)
        x207=self.dropout1(x206)
        x208=self.conv2d62(x207)
        return x208

m = M().eval()
x204 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x204)
end = time.time()
print(end-start)
