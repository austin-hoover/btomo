%----------------------------
% This program is used to generate initial uniform distribution
%--------------------------------------------
clear
d=dlmread('beam_distr_after_Q1.txt','',3,0);
N=5000000; % number of particles
Mass=939.294; %Mass(MeV)
Beam_energy= 6.8; %MeV
Beam_f =325; %MHz
Beam_I= 100; %mA
Beam_Charge=-1;
range_x=20;%分布半包络
range_xp=10;%分布半张角
range_y=20;%分布半包络
range_yp=15;%分布半张角
stepx=sqrt((2*range_x+1)*(2*range_xp+1)/N);
stepy=sqrt((2*range_y+1)*(2*range_yp+1)/N);
x=zeros(N,1);
xp=zeros(N,1);
y=zeros(N,1);
yp=zeros(N,1);
% z=zeros(N,1);
% zp=zeros(N,1);
% p=zeros(N,1);
% t=zeros(N,1);
% e=zeros(N,1);
% l=zeros(N,1);
Nx=ceil((2*range_x+1)/stepx);
Nxp=ceil((2*range_xp+1)/stepx);
Ny=ceil((2*range_y+1)/stepy);
Nyp=ceil((2*range_y+1)/stepy);

    
 x=unifrnd(-range_x,range_x,N,1);
 xp=unifrnd(-range_xp,range_xp,N,1);
 y=unifrnd(-range_y,range_y,N,1);
 yp=unifrnd(-range_yp,range_yp,N,1);
       

in=find(d(:,10)~=0|d(:,9)<2);% 去掉计算结果中丢掉的粒子和未被加速的粒子
d(in,:)=[];
countnum=ceil(N/length(d(:,1)));
d=repmat(d,countnum,1);
  z=d(1:N,5);
  zp=d(1:N,6);
  p=d(1:N,7);
  t=d(1:N,8);
  e=d(1:N,9);
  l=d(1:N,10);
  data=[x,xp,y,yp,z,zp,p,t,e,l];
   pdit=zeros(N,6);
   pdit(:,[1,3])=data(:,[1,3])/10;
   pdit(:,[2,4])=data(:,[2,4])/1000;
   pdit(:,5)=data(:,7)/180*pi;
   pdit(:,6)=data(:,9);
 txt2tracewindst(pdit,0,Beam_I,Mass);
   save parnum.mat N
  