%exp ------------ 4

f=rand(100,1) %sampling from uniform dist
subplot(2,2,1)
histogram(f,'Normalization','probability')
title('U1 Uni dist')

g=rand(100,1) %sampling from uniform dist
subplot(2,2,3)
histogram(g,'Normalization','probability')
title('U2 Uni dist')

w=zeros(100,1)
e=zeros(100,1)
for i=1:100 %box muller trasform
    w(i)=sqrt(-2*log(f(i)))*cos(2*pi*g(i))
    e(i)=sqrt(-2*log(f(i)))*sin(2*pi*g(i))
end

subplot(2,2,2)
histogram(w,'Normalization','probability')
title('Normal Z0')

subplot(2,2,4)
histogram(e,'Normalization','probability')
title('Normal Z1')
