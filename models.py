import torch

MIN_CHANNELS = 32


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int,k_s=3,stride=1,padding=1) -> None:
        super(ConvBlock, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_s, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SingleScaleGenerator(torch.nn.Module):

    def __init__(self, n_channels: int = 32, min_channels: int = MIN_CHANNELS, n_blocks=5) -> None:
        super(SingleScaleGenerator, self).__init__()

        self.head = ConvBlock(in_channels=3, out_channels=n_channels)

        self.body = torch.nn.ModuleList()
        for i in range(n_blocks-2):
            in_channels = max([min_channels, n_channels // (2 ** (i))])
            out_channels = max([min_channels, n_channels // (2 ** (i+1))])
            self.body.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
        self.body = torch.nn.Sequential(*self.body)

        self.tail = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        r = x + z
        r = self.tail(self.body(self.head(r)))
        return x + r
    # def forward(self,x,y):
    #     x = self.head(x)
    #     x = self.body(x)
    #     x = self.tail(x)
    #     ind = int((y.shape[2]-x.shape[2])/2)
    #     y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
    #     return x+y


class Discriminator(torch.nn.Module):

    def __init__(self, n_channels: int = 32, min_channels: int = MIN_CHANNELS, n_blocks=5,head_ks=3) -> None:
        super(Discriminator, self).__init__()

        self.head = ConvBlock(in_channels=3, out_channels=n_channels,k_s=head_ks)

        self.body = torch.nn.ModuleList()
        for i in range(n_blocks-2):
            in_channels = max([min_channels, n_channels // (2 ** (i))])
            out_channels = max([min_channels, n_channels // (2 ** (i+1))])
            self.body.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
        self.body = torch.nn.Sequential(*self.body)

        self.tail = torch.nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=3, stride=1, padding=1)


    def forward(self, x) -> None:
        return self.tail(self.body(self.head(x)))


## For sinkhorn loss the discriminator will need many output channels.
## We change the kernel size dynamically otherwise the batch size gets too large.
class Discriminator_sk(torch.nn.Module):

    def __init__(self, n_channels: int = 32, min_channels: int = MIN_CHANNELS, n_blocks=5,tail_ker_s=5,tail_stride=1) -> None:
        super(Discriminator_sk, self).__init__()

        self.head = ConvBlock(in_channels=3, out_channels=n_channels,k_s=3,stride=1,padding=1)

        self.body = torch.nn.ModuleList()
        for i in range(n_blocks-2):
            in_channels = max([min_channels, n_channels // (2 ** (i))])
            out_channels = max([min_channels, n_channels // (2 ** (i+1))])
            self.body.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
        self.body = torch.nn.Sequential(*self.body)

        self.tail = torch.nn.Conv2d(in_channels=out_channels, out_channels=32, kernel_size=tail_ker_s, stride=tail_stride, padding=tail_ker_s//2)


    def forward(self, x) -> None:
        #return self.tail(self.body(self.head(x)))
        x = self.tail(self.body(self.head(x)))
        out_shape = x.size()
        return x.reshape((x.shape[1],-1)).T ## Size (n_patches,n_channels)
    
    def output_size(self,x):
        x = self.tail(self.body(self.head(x)))
        out_shape = x.size()
        return out_shape
        
        

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)