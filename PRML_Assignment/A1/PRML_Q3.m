%exp ------- 3
% part 1: implementing CLT from uni dist samples [0,1]
u=rand(100) %creates 100x100 matrix with samples from uniform dist [0,1] : each row itself represents samples from uni dist [0,1]

subplot(2,2,1)
histogram(u(32,:),'Normalization','probability') % creating histogram of any arbitary row of 'u' to show that it follows uni dist [0,1]
%each row of 'u' represents a unique random uniform variable samples [0,1]
title('arbitary uni dist')

s=sum(u) %returns a row vector with sum of each column
% entres of 's' are summation of various uni dist [0,1] RV drawn from columns of 'u'

q=s'
sum1=0
sum2=0

subplot(2,2,2)
histogram(q,'Normalization','probability') % creating histogram to show that each point in 's' represents a smaple from normal distribution
title('Normal obtained from CLT')

% mean of 'u' is 0.5
%var of 'u' is 1/12
% acc to CLT 'u'~N(100*mean of'u',100*var of 'u')
% we check weather mean and var of 's' are 50, 8.33 respectevely
for k=1:100 %mean finder
    sum1=sum1+q(k)
    
end

d=sum1/100 %mean of 's'

for m=1:100
    l=sum2+((q(m)-d)*(q(m)-d))
end

v=l/100 %var of 's'

% part 2


m1=rand(12,100) %each row herte is a uni RV with [0.1]

subplot(2,2,3)
histogram(m1(12,:),'Normalization','probability') %proof that each row rep a uni RV
title('arbitary Uni dist')

s1=sum(m1)

c1=s1'



for ue=1:100
    c1(ue)=c1(ue)/12
end

subplot(2,2,4)
histogram(c1,'Normalization','probability')
title('PDF of sample mean')

%means is same, can be seen from histogram plotted, hence proved.
