a = sqrt(20);
b = 10;
y = a.*randn(200,1) + b; %sampling from N(10,10)

c = 5
d = 20
z = c.*randn(200,1) + b; %sampling from N(20,25)

R=unifrnd(5,20,40,1) %sampling from U[5,20]

fx=zeros(40,1)
fy=zeros(40,1)

%getting pdf values from data points from U[5,20] of both nromal
%distribution respectevely fx, fy
for i=1:40
    
    fx(i)=((1/a*sqrt(2*pi)))*(exp(-0.5*(((R(i)-b)*(R(i)-b))/(a*a))))
    fy(i)=((1/c*sqrt(2*pi)))*(exp(-0.5*(((R(i)-d)*(R(i)-d))/(c*c))))
end
%------------------if point belongs to first normal; 1 is assinged else 0

% for prior (0.5,0.5)
class1=zeros(40,1)
for j=1:40
    if(0.5*fx(j) > 0.5*fy(j))
        class1(j)=1
    else
        class1(j)=0
    end
end

%for prior (0.7,0.3)
class2=zeros(40,1)
for k=1:40
    if(0.7*fx(k) > 0.3*fy(k))
        class2(k)=1
    else
        class2(k)=0
    end
end

%for prior (0.3,0.7)
class3=zeros(40,1)
for l=1:40
    if(0.3*fx(l) > 0.7*fy(l))
        class3(l)=1
    else
        class3(l)=0
    end
end