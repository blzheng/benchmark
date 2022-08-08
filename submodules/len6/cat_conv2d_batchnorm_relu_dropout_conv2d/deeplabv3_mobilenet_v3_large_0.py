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
        self.conv2d67 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d51 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu24 = ReLU()
        self.dropout0 = Dropout(p=0.5, inplace=False)
        self.conv2d68 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x186, x189, x192, x195, x202):
        x203=torch.cat([x186, x189, x192, x195, x202],dim=1)
        x204=self.conv2d67(x203)
        x205=self.batchnorm2d51(x204)
        x206=self.relu24(x205)
        x207=self.dropout0(x206)
        x208=self.conv2d68(x207)
        return x208

m = M().eval()
x186 = torch.randn(torch.Size([1, 256, 14, 14]))
x189 = torch.randn(torch.Size([1, 256, 14, 14]))
x192 = torch.randn(torch.Size([1, 256, 14, 14]))
x195 = torch.randn(torch.Size([1, 256, 14, 14]))
x202 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x186, x189, x192, x195, x202)
end = time.time()
print(end-start)
