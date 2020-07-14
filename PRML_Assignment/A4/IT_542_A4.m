% (1) Sampling
a=sqrt(10);
b=5;

p1=a.*randn(1000,1)+b; % p1 N(5,10)

c=sqrt(15);
d=10;

p2=c.*randn(1000,1)+d; %p2 N(10,15)

P1=0.6;
P2=0.4;

p=P1*p1+P2*p2;

histogram(p,'Normalization','probability')

S=std(p);

h=1.06*S*0.2511;

% (2) KDE Estimate
figure
pdSix = fitdist(p,'Kernel','BandWidth',h);
x = 0:0.1:20;
ySix = pdf(pdSix,x);
plot(x,ySix,'k-','LineWidth',2)
