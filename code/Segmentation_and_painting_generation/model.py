nclasses = 9 

class my_FCN(nn.Module):
    def __init__(self, pretrained_model):
        super(my_FCN, self).__init__()
        self.backbone = torch.nn.Sequential(*list(pretrained_model.children())[:-2])
        
        self.Classifer =torch.nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(512, 9, 1))
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        x = self.backbone(x)['out']
        x = self.Classifer(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x