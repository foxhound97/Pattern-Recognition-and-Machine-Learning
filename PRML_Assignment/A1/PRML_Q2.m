% exp ------ 2

% for 100 data points
mu1 = [4 6];
sigma1 = [1 1.5; 1.5 3];
Rnorm1 = mvnrnd(mu1,sigma1,100); %sampling from bvariate normal for 100 points

y1 = mvnpdf(Rnorm1,mu1,sigma1)

rho4=corr(Rnorm1,y1) %comparing bivariate normal RV

%plotting function
subplot(3,1,1) %fig 1
scatter3(Rnorm1(:,1),Rnorm1(:,2),y1)
xlabel('X1')
ylabel('X2')
zlabel('Probability Density')
title('Fig 1')


%for 500 data points
mu2 = [4 6];
sigma2 = [1 1.5; 1.5 3];
Rnorm2 = mvnrnd(mu2,sigma2,500);%sampling from bvariate normal for 500 points
y2 = mvnpdf(Rnorm2,mu2,sigma2)
rho5=corr(Rnorm2,y2)

subplot(3,1,2) %fig 2
scatter3(Rnorm2(:,1),Rnorm2(:,2),y2)
xlabel('X1')
ylabel('X2')
zlabel('Probability Density')
title('Fig 2')

% for 1000 data points
mu3 = [4 6];
sigma3 = [1 1.5; 1.5 3];
Rnorm3 = mvnrnd(mu3,sigma3,1000);%sampling from bvariate normal for 1000 points
y3 = mvnpdf(Rnorm3,mu3,sigma3)
rho6=corr(Rnorm3,y3)

subplot(3,1,3) %fig 2
scatter3(Rnorm3(:,1),Rnorm3(:,2),y3)
xlabel('X1')
ylabel('X2')
zlabel('Probability Density')
title('Fig 3')

