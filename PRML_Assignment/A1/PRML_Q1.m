% exp ----- 1

% for 100 data points
a=randn(100,1) % samples from normal distribution

subplot(3,2,1) %fig 1
histogram(a,'Normalization','probability') % creating histogram 
title('histogram of samples drawn from uni variate normal for 100 points')

subplot(3,2,2) %fig 2
y=normpdf(a) % computing pdf
plot(a,y)
title('pdf generated from the samles drawn for 100 points')

rho1 = corr(a,y) %comparing two vectors; linear correlation

%--------------------------------------------------------------------------------

%for 500 data points

b=randn(500,1) % samples from normal distribution

subplot(3,2,3) %fig 3
histogram(b,'Normalization','probability') % creating histogram 
title('histogram of samples drawn from uni variate normal for 500 points')

subplot(3,2,4) %fig 4
z=normpdf(b) % computing pdf
plot(b,z)
title('pdf generated from the samles drawn for 500 points')

rho2=corr(b,z)

%------------------------------------------------------------------------------------

% for 1000 data points

c=randn(1000,1) % samples from normal distribution

subplot(3,2,5) %fig 5
histogram(c,'Normalization','probability') % creating histogram 
title('histogram of samples drawn from uni variate normal for 1000 points')

subplot(3,2,6) %fig 6
x=normpdf(c) % computing pdf
plot(c,x)
title('pdf generated from the samles drawn for 1000 points')

rho3=corr(c,x)




