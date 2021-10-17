# ConvMixer: Patches Are All You Need?
This repo comes with an implementation of ConvMixer for the ICLR 2022 submission ["Patches Are All You Need?"](https://openreview.net/forum?id=TVHS5Y4dNvM).

The official implementation is very very short, to be specific only just 280 characters. All you need is just to use “from torch.nn import * ” before the model:

```
def ConvMixr(h,d,k,p,n):
 S,C,A=Sequential,Conv2d,lambda x:S(x,GELU(),BatchNorm2d(h))
 R=type('',(S,),{'forward':lambda s,x:s[0](x)+x})
 return S(A(C(3,h,p,p)),*[S(R(A(C(h,h,k,groups=h,padding=k//2))),A(C(h,h,1))) for i in range(d)],AdaptiveAvgPool2d((1,1)),Flatten(),Linear(h,n))
```