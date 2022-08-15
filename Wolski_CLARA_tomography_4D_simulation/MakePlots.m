clear

close all

load PhaseSpaceDensityMeasured
load PhaseSpaceDensity
% load CovarianceMatrix

% betax        =  6.4911;
% alphax       = -1.3239;
% betay        =  1.4290;
% alphay       = -1.9192;

betax        = Beta_x_y_at_reconstruction_point(1);
alphax       = Alpha_x_y_at_reconstruction_point(1);
betay        = Beta_x_y_at_reconstruction_point(2);
alphay       = Alpha_x_y_at_reconstruction_point(2);

relgamma      = BeamMomentum/0.511;

%--------------------------------------------------------------------------
% Calculate the normal mode emittances

intnorm       = sum(sum(sum(sum(rho))));

maxx          = calibn1(1)*(size(rho,1)-1)/2;
maxpx         = calibn1(1)*(size(rho,2)-1)/2;
maxy          = calibn1(2)*(size(rho,3)-1)/2;
maxpy         = calibn1(2)*(size(rho,4)-1)/2;

rhox          = reshape(sum(sum(sum(rho,4),3),2),[1,49]);
range1        = -maxx:(2*maxx/(size(rho,1)-1)):maxx;
meanx2        = sum(range1.*range1.*rhox)/intnorm;

rhopx         = reshape(sum(sum(sum(rho,4),3),1),[1,49]);
range1        = -maxpx:(2*maxpx/(size(rho,2)-1)):maxpx;
meanpx2       = sum(range1.*range1.*rhopx)/intnorm;

rhoy          = reshape(sum(sum(sum(rho,4),2),1),[1,49]);
range1        = -maxy:(2*maxy/(size(rho,3)-1)):maxy;
meany2        = sum(range1.*range1.*rhoy)/intnorm;

rhopy         = reshape(sum(sum(sum(rho,3),2),1),[1,49]);
range1        = -maxpy:(2*maxpy/(size(rho,4)-1)):maxpy;
meanpy2       = sum(range1.*range1.*rhopy)/intnorm;

rhoxpx        = squeeze(sum(sum(rho,4),3))';
range1        = -maxx :(2*maxx /(size(rho,1)-1)):maxx;
range2        = -maxpx:(2*maxpx/(size(rho,2)-1)):maxpx;
[range1, range2] = meshgrid(range1, range2);
meanxpx       = sum(sum(range1.*range2.*rhoxpx))/intnorm;

rhoxy         = squeeze(sum(sum(rho,4),2))';
range1        = -maxx:(2*maxx/(size(rho,1)-1)):maxx;
range2        = -maxy:(2*maxy/(size(rho,3)-1)):maxy;
[range1, range2] = meshgrid(range1, range2);
meanxy        = sum(sum(range1.*range2.*rhoxy))/intnorm;

rhoxpy        = squeeze(sum(sum(rho,3),2))';
range1        = -maxx :(2*maxx /(size(rho,1)-1)):maxx;
range2        = -maxpy:(2*maxpy/(size(rho,4)-1)):maxpy;
[range1, range2] = meshgrid(range1, range2);
meanxpy       = sum(sum(range1.*range2.*rhoxpy))/intnorm;

rhopxy        = squeeze(sum(sum(rho,4),1))';
range1        = -maxpx:(2*maxpx/(size(rho,2)-1)):maxpx;
range2        = -maxy :(2*maxy /(size(rho,3)-1)):maxy;
[range1, range2] = meshgrid(range1, range2);
meanpxy       = sum(sum(range1.*range2.*rhopxy))/intnorm;

rhopxpy       = squeeze(sum(sum(rho,3),1))';
range1        = -maxpx:(2*maxpx/(size(rho,2)-1)):maxpx;
range2        = -maxpy:(2*maxpy/(size(rho,4)-1)):maxpy;
[range1, range2] = meshgrid(range1, range2);
meanpxpy      = sum(sum(range1.*range2.*rhopxpy))/intnorm;

rhoypy        = squeeze(sum(sum(rho,2),1))';
range1        = -maxy :(2*maxy /(size(rho,3)-1)):maxy;
range2        = -maxpy:(2*maxpy/(size(rho,4)-1)):maxpy;
[range1, range2] = meshgrid(range1, range2);
meanypy       = sum(sum(range1.*range2.*rhoypy))/intnorm;

sigmamat      = [meanx2   meanxpx   meanxy   meanxpy;...
                 meanxpx  meanpx2   meanpxy  meanpxpy;...
                 meanxy   meanpxy   meany2   meanypy;...
                 meanxpy  meanpxpy  meanypy  meanpy2];

smat          = [ 0  1  0  0;...
                 -1  0  0  0;...
                  0  0  0  1;...
                  0  0 -1  0];
              
[evec, emit] = eig(sigmamat*smat);
[emit, ordr] = sort(diag(imag(emit)));

evec = evec(:, ordr);

t1 = diag([1i 0 0 -1i]);
t2 = diag([0 1i -1i 0]);

invnrm       = [ sqrt(betax)        0             0                    0;...
                -alphax/sqrt(betax) 1/sqrt(betax) 0                    0;...
                 0                  0             sqrt(betay)          0;...
                 0                  0            -alphay/sqrt(betay)   1/sqrt(betay)];

beta1 = real(invnrm*evec*t1/evec*smat*invnrm');
beta2 = real(invnrm*evec*t2/evec*smat*invnrm');

fileID = fopen('Optics.txt','w');

fprintf(fileID,'Normalised emittances:\r\n');
fprintf(fileID,'  Mode 1: %10.6f um\r\n',    relgamma*emit(3));
fprintf(fileID,'  Mode 2: %10.6f um\r\n\r\n',relgamma*emit(4));

fprintf(fileID,'Mode 1 Beta matrix:\r\n');
fprintf(fileID,'  %10.6f %10.6f %10.6f %10.6f \r\n',  beta1);

fprintf(fileID,'\r\n');
fprintf(fileID,'Mode 2 Beta matrix:\r\n');
fprintf(fileID,'  %10.6f %10.6f %10.6f %10.6f \r\n',  beta2);

fclose(fileID);

type('Optics.txt')

%--------------------------------------------------------------------------

fignum1 = 3;
fignum2 = 4;

% close all

alfxrp = 0;
betxrp = 1;  

alfyrp = 0;
betyrp = 1;

psresn = size(rho,1);

psresnx  = psresn*sqrt(betxrp);
psresnpx = psresn*sqrt((1+alfxrp^2)/betxrp);
psresny  = psresn*sqrt(betyrp);
psresnpy = psresn*sqrt((1+alfyrp^2)/betyrp);

rho1   = zeros(round(psresnx),round(psresnpx),round(psresny),round(psresnpy));

for xi = -psresnx/2:psresnx/2
    
    xin = xi/sqrt(betxrp) + (psresn+1)/2;
   
    for pxi = -psresnpx/2:psresnpx/2
        
        pxin = alfxrp*xi/sqrt(betxrp) + sqrt(betxrp)*pxi  + (psresn+1)/2;
       
        for yi = -psresny/2:psresny/2
           
            yin = yi/sqrt(betyrp) + (psresn+1)/2;
            
            for pyi = -psresnpy/2:psresnpy/2
                
                pyin = alfyrp*yi/sqrt(betyrp) + sqrt(betyrp)*pyi  + (psresn+1)/2;
                
                rval = 0;
                
                if xin>1 && xin<psresn...
                        && pxin>1 && pxin<psresn...
                        && yin>1  && yin<psresn...
                        && pyin>1 && pyin<psresn
                    
                xin0  = floor(xin);  xin1  = xin0+1;  xind  = xin - xin0;    
                yin0  = floor(yin);  yin1  = yin0+1;  yind  = yin - yin0;
                pxin0 = floor(pxin); pxin1 = pxin0+1; pxind = pxin - pxin0;    
                pyin0 = floor(pyin); pyin1 = pyin0+1; pyind = pyin - pyin0;
                    
                rval0000 = rho(xin0,pxin0,yin0,pyin0);
                rval1000 = rho(xin1,pxin0,yin0,pyin0);
                rval0010 = rho(xin0,pxin0,yin1,pyin0);
                rval1010 = rho(xin1,pxin0,yin1,pyin0);
                    
                rval00   = rval0000*(1-xind)*(1-yind) + ...
                           rval1000*xind*(1-yind) + ...
                           rval0010*(1-xind)*yind + ...
                           rval1010*xind*yind;
                     
                rval0100 = rho(xin0,pxin1,yin0,pyin0);
                rval1100 = rho(xin1,pxin1,yin0,pyin0);
                rval0110 = rho(xin0,pxin1,yin1,pyin0);
                rval1110 = rho(xin1,pxin1,yin1,pyin0);
                    
                rval10   = rval0100*(1-xind)*(1-yind) + ...
                           rval1100*xind*(1-yind) + ...
                           rval0110*(1-xind)*yind + ...
                           rval1110*xind*yind;
                
                rval0001 = rho(xin0,pxin0,yin0,pyin1);
                rval1001 = rho(xin1,pxin0,yin0,pyin1);
                rval0011 = rho(xin0,pxin0,yin1,pyin1);
                rval1011 = rho(xin1,pxin0,yin1,pyin1);
                    
                rval01   = rval0001*(1-xind)*(1-yind) + ...
                           rval1001*xind*(1-yind) + ...
                           rval0011*(1-xind)*yind + ...
                           rval1011*xind*yind;
                     
                rval0101 = rho(xin0,pxin1,yin0,pyin1);
                rval1101 = rho(xin1,pxin1,yin0,pyin1);
                rval0111 = rho(xin0,pxin1,yin1,pyin1);
                rval1111 = rho(xin1,pxin1,yin1,pyin1);
                    
                rval11   = rval0101*(1-xind)*(1-yind) + ...
                           rval1101*xind*(1-yind) + ...
                           rval0111*(1-xind)*yind + ...
                           rval1111*xind*yind;
                     
                rval     = rval00*(1-pxind)*(1-pyind) + ...
                           rval10*pxind*(1-pyind) + ...
                           rval01*(1-pxind)*pyind + ...
                           rval11*pxind*pyind;

                end
                
                rho1(round(xi+(psresnx+2)/2),...
                     round(pxi+(psresnpx+2)/2),...
                     round(yi+(psresny+2)/2),...
                     round(pyi+(psresnpy+2)/2)) = ...
                                      rval;
                                  
            end
            
        end
        
    end
    
end


sigmat = [betxrp 0        0      0        ;...
          0      1/betxrp 0      0        ;...
          0      0        betyrp 0        ;...
          0      0        0      1/betyrp];

h        = fspecial('average',[2 2]);
% nptcles  = 1:1000;

% Set the screen calibration values.
% The factor 1.2 comes from resizing the cropped camera image
% to match the desired phase space resolution (see CLARATomography script).
calibnx  = 0.0122*1.15;  % millimetres/pixel
calibny  = 0.0122*1.15;  % millimetres/pixel

rangex   = calibnx*(size(rho1,1)+1);
rangepx  = calibnx*(size(rho1,2)+1);
rangey   = calibny*(size(rho1,3)+1);
rangepy  = calibny*(size(rho1,4)+1);

zoomx    =  1:size(rho1,1);
zoompx   =  1:size(rho1,2);
zoomy    =  1:size(rho1,3);
zoompy   =  1:size(rho1,4);

smatrix = [ 0  1  0  0;...
           -1  0  0  0;...
            0  0  0  1;...
            0  0 -1  0];
        
scalemat = diag([calibnx calibnx calibny calibny]);
        
emit  = abs(imag(eig(scalemat*sigmat*scalemat*smatrix)));

emitxn = 8.86*min(emit);
emityn = 8.86*max(emit);

figure(fignum1)

phi   = 0:(2*pi/100):2*pi;

% rho1 = rho; %permute(rho,[4 3 2 1]);

%--------------------------------------------------------------------------
% Plot the horizontal phase space distribution

rhoxpx  = sum(sum(rho1,4),3);
rhoxpx  = rhoxpx(zoomx,zoompx);
rhoxpxf = imfilter(rhoxpx,h);

subplot(3,2,1)
hold off
imagesc([-rangex rangex],[-rangepx rangepx],rhoxpxf');
set(gca,'YDir','normal')
hold on

% jval  = 10; %sqrt(det(twiss));
% 
% xvals  = calibnx*sqrt(2*betxrp*jval)*cos(phi);
% pxvals =-calibnx*sqrt(2*jval/betxrp)*(sin(phi) + alfxrp*cos(phi));
% 
% plot(xvals,pxvals,'-k')

% axis equal
axis tight
xlabel('x_N (mm/\surdm)')
ylabel('p_{xN} (mm/\surdm)')
% title(['\gamma\epsilon_x = ' num2str(emitxn,'%1.2f') ' \mum'])

%--------------------------------------------------------------------------
rhoxpy  = permute(rho1,[1,4,2,3]);
rhoxpy  = sum(sum(rhoxpy,4),3);
rhoxpy  = rhoxpy(zoomx,zoompy);
rhoxpyf = imfilter(rhoxpy,h);

subplot(3,2,5)
imagesc([-rangex rangex],[-rangepy rangepy],rhoxpyf');
set(gca,'YDir','normal')
hold on

% twiss = [sigmat(1,1) sigmat(1,4);...
%          sigmat(4,1) sigmat(4,4)];
%      
% jval  = sqrt(det(twiss));
% 
% beta  = twiss(1,1)/jval;
% alpha =-twiss(1,2)/jval;
% 
% xvals  = calibnx*sqrt(2*beta*jval)*cos(phi);
% pyvals =-calibny*sqrt(2*jval/beta)*(sin(phi) + alpha*cos(phi));
% 
% plot(xvals,pyvals,'-k')

% axis equal
axis tight
xlabel('x_N (mm/\surdm)')
ylabel('p_{yN} (mm/\surdm)')

%--------------------------------------------------------------------------
rhoypy  = permute(rho1,[3,4,1,2]);
rhoypy  = sum(sum(rhoypy,4),3);
rhoypy  = rhoypy(zoomy,zoompy);
rhoypyf = imfilter(rhoypy,h);

subplot(3,2,2)
imagesc([-rangey rangey],[-rangepy rangepy],rhoypyf');
set(gca,'YDir','normal')
hold on

% twiss = [sigmat(3,3) sigmat(3,4);...
%          sigmat(4,3) sigmat(4,4)];
%      
% jval  = sqrt(det(twiss));
% 
% beta  = twiss(1,1)/jval;
% alpha =-twiss(1,2)/jval;
% 
% yvals  = calibny*sqrt(2*beta*jval)*cos(phi);
% pyvals =-calibny*sqrt(2*jval/beta)*(sin(phi) + alpha*cos(phi));

% plot(yvals,pyvals,'-k')

% axis equal
axis tight
xlabel('y_N (mm/\surdm)')
ylabel('p_{yN} (mm/\surdm)')
% title(['\gamma\epsilon_y = ' num2str(emityn,'%1.2f') ' \mum'])

%--------------------------------------------------------------------------
rhoypx  = permute(rho1,[3,2,1,4]);
rhoypx  = sum(sum(rhoypx,4),3);
rhoypx  = rhoypx(zoomy,zoompx);
rhoypxf = imfilter(rhoypx,h);

subplot(3,2,6)
imagesc([-rangey rangey],[-rangepx rangepx],rhoypxf');
set(gca,'YDir','normal')
hold on

% twiss = [sigmat(3,3) sigmat(3,2);...
%          sigmat(2,3) sigmat(2,2)];
%      
% jval  = sqrt(det(twiss));
% 
% beta  = twiss(1,1)/jval;
% alpha =-twiss(1,2)/jval;
% 
% yvals  = calibny*sqrt(2*beta*jval)*cos(phi);
% pxvals =-calibnx*sqrt(2*jval/beta)*(sin(phi) + alpha*cos(phi));
% 
% plot(yvals,pxvals,'-k')

% axis equal
axis tight
xlabel('y_N (mm/\surdm)')
ylabel('p_{xN} (mm/\surdm)')

%--------------------------------------------------------------------------
rhoxy   = permute(rho1,[1,3,2,4]);
rhoxy   = sum(sum(rhoxy,4),3);
rhoxy   = rhoxy(zoomx,zoomy);
rhoxyf  = imfilter(rhoxy,h);

subplot(3,2,3)
imagesc([-rangex rangex],[-rangey rangey],rhoxyf');
set(gca,'YDir','normal')
hold on

% twiss = [sigmat(1,1) sigmat(1,3);...
%          sigmat(3,1) sigmat(3,3)];
%      
% jval  = sqrt(det(twiss));
% 
% beta  = twiss(1,1)/jval;
% alpha =-twiss(1,2)/jval;
% 
% xvals  = calibnx*sqrt(2*beta*jval)*cos(phi);
% yvals  =-calibny*sqrt(2*jval/beta)*(sin(phi) + alpha*cos(phi));
% 
% plot(xvals,yvals,'-k')

% axis equal
axis tight
xlabel('x_N (mm/\surdm)')
ylabel('y_N (mm/\surdm)')

%--------------------------------------------------------------------------

rhopxpy  = permute(rho1,[2,4,1,3]);
rhopxpy  = sum(sum(rhopxpy,4),3);
rhopxpy  = rhopxpy(zoompx,zoompy);
rhopxpyf = imfilter(rhopxpy,h);

subplot(3,2,4)
imagesc([-rangepx rangepx],[-rangepy rangepy],rhopxpyf');
set(gca,'YDir','normal')
hold on

% twiss = [sigmat(2,2) sigmat(2,4);...
%          sigmat(4,2) sigmat(4,4)];
%      
% jval  = sqrt(det(twiss));
% 
% beta  = twiss(1,1)/jval;
% alpha =-twiss(1,2)/jval;
% 
% pxvals = calibnx*sqrt(2*beta*jval)*cos(phi);
% pyvals =-calibny*sqrt(2*jval/beta)*(sin(phi) + alpha*cos(phi));
% 
% plot(pxvals,pyvals,'-k')

% axis equal
axis tight
xlabel('p_{xN} (mm/\surdm)')
ylabel('p_{yN} (mm/\surdm)')

set(gcf,'PaperUnits','inches')
set(gcf,'PaperPosition',[1 1 6 10])
print('-dpng','Tomography4DPhaseSpace-Iteration2.png','-r600')
print('-dpdf','Tomography4DPhaseSpace-Iteration2.pdf')

return

%--------------------------------------------------------------------------

figure(fignum2)
subplot(2,2,3)
imagesc([-rangex rangex],[-rangey rangey],rhoxyf');
set(gca,'YDir','normal')
axis([-rangex rangex -20 20])
xlabel('x (mm)')
ylabel('y (mm)')
title('Reconstructed image at Screen 2')

xprojection = sum(rhoxy(zoomx,zoomy),2);
xvals       = ((1:size(xprojection,1))-size(xprojection,1)/2)*2*rangex/size(xprojection,1);
subplot(2,2,1)
plot(xvals,xprojection,'-k')
hold on
sigx1 = 12*sqrt(betxrp)*calibnx;
plot(xvals,max(xprojection)*exp(-(xvals/sigx1).^2/2),'--r')
axis([-rangex rangex -inf inf])

yprojection = sum(rhoxy(zoomx,zoomy),1)';
yvals       = ((1:size(yprojection,1))-size(yprojection,1)/2)*2*rangey/size(yprojection,1);
subplot(2,2,4)
plot(yprojection,yvals,'-k')
hold on
sigy1 = 6*sqrt(betyrp)*calibny;
plot(max(yprojection)*exp(-(yvals/sigy1).^2/2),yvals,'--r')
axis([-inf inf -20 20])


return


[~, ~, twiss1] = DefineBeamline();

betax1   = twiss1(1,1,1);
alphax1  =-twiss1(1,2,1);

betay1   = twiss1(3,3,2);
alphay1  =-twiss1(3,4,2);

binsxnrm = binsx/sqrt(betax1);
binsynrm = binsy/sqrt(betay1);
    
nrmx     = [      1/sqrt(betax1)      0       ;...
            alphax1/sqrt(betax1) sqrt(betax1) ];

psNorm   = nrmx*(particles(1:2,nptcles));        
        
pstx     = TransformImage(rhoxpx,inv(nrmx));
pstxf    = imfilter(pstx,h);

pstscale = nrmx\[ min(binsxnrm) max(binsxnrm) ;...
                  min(binsxnrm) max(binsxnrm) ];

figure(1)

subplot(3,4,1)
plot(psNorm(1,nptcles)*1e3,psNorm(2,nptcles)*1e3,'.k','MarkerSize',1)
axis equal
axis(1e3*[min(binsxnrm) max(binsxnrm) min(binsxnrm) max(binsxnrm)])
xlabel('x_N')
ylabel('p_{x,N}')

subplot(3,4,2)
imagesc(1e3*[min(binsxnrm) max(binsxnrm)],1e3*[min(binsxnrm) max(binsxnrm)],rhoxpxf')
set(gca,'YDir','normal')
axis equal
axis(1e3*[min(binsxnrm) max(binsxnrm) min(binsxnrm) max(binsxnrm)])
xlabel('x_N')
ylabel('p_{x,N}')

subplot(3,4,3)
plot(particles(1,nptcles)*1e3,particles(2,nptcles)*1e3,'.k','MarkerSize',1)
axis square
axis([-2 2 -2 2])
psaxis = axis;
xlabel('x (mm)')
ylabel('p_x (10^{-3})')

subplot(3,4,4)
imagesc(1e3*[pstscale(1,1) pstscale(1,2)],1e3*[pstscale(2,1) pstscale(2,2)],pstxf')
set(gca,'YDir','normal')
axis equal
axis(psaxis)
xlabel('x (mm)')
ylabel('p_x (10^{-3})')

% figure(3)
% ivector1 = dfull*rhoxpx;
% plot(xyprojection,'-k')
% hold on
% plot(ivector1,'--r')

% Plot the vertical phase space distribution

rhoypy = permute(rho,[3,4,1,2]);
rhoypy = sum(sum(rhoypy,4),3);
rhoypyf  = imfilter(rhoypy,h);

nrmy     = [      1/sqrt(betay1)      0       ;...
            alphay1/sqrt(betay1) sqrt(betay1) ];

psNorm   = nrmx*(particles(3:4,nptcles));

psty     = TransformImage(rhoypy,inv(nrmy));
pstyf    = imfilter(psty,h);

pstscale = nrmy\[ min(binsynrm) max(binsynrm) ;...
                  min(binsynrm) max(binsynrm) ];

% figure(2)

subplot(3,4,5)
plot(psNorm(1,nptcles)*1e3,psNorm(2,nptcles)*1e3,'.k','MarkerSize',1)
axis equal
axis(1e3*[min(binsxnrm) max(binsxnrm) min(binsxnrm) max(binsxnrm)])
xlabel('y_N')
ylabel('p_{y,N}')

subplot(3,4,6)
imagesc(1e3*[min(binsynrm) max(binsynrm)],1e3*[min(binsynrm) max(binsynrm)],rhoypyf')
set(gca,'YDir','normal')
axis equal
axis(1e3*[min(binsynrm) max(binsynrm) min(binsynrm) max(binsynrm)])
xlabel('y_N')
ylabel('p_{y,N}')

subplot(3,4,7)
plot(particles(3,nptcles)*1e3,particles(4,nptcles)*1e3,'.k','MarkerSize',1)
axis square
axis([-2 2 -2 2])
psaxis = axis;
xlabel('y (mm)')
ylabel('p_y (10^{-3})')

subplot(3,4,8)
imagesc(1e3*[pstscale(1,1) pstscale(1,2)],1e3*[pstscale(2,1) pstscale(2,2)],pstyf')
set(gca,'YDir','normal')
axis equal
axis(psaxis)
xlabel('y (mm)')
ylabel('p_y (10^{-3})')

% Plot the co-ordinate phase space distribution

rhoxy    = permute(rho,[1,3,2,4]);
rhoxy    = sum(sum(rhoxy,4),3);
rhoxyf   = imfilter(rhoxy,h);

nrmxy    = [      1/sqrt(betax1)  0               ;...
                  0               1/sqrt(betay1) ];

csNorm   = nrmxy*(particles([1 3],nptcles));

pstxy    = TransformImage(rhoxy,inv(nrmxy));
pstxyf   = imfilter(pstxy,h);

pstscale = nrmxy\[ min(binsxnrm) max(binsxnrm) ;...
                   min(binsynrm) max(binsynrm) ];

% figure(3)

subplot(3,4,9)
plot(csNorm(1,nptcles)*1e3,csNorm(2,nptcles)*1e3,'.k','MarkerSize',1)
axis equal
axis(1e3*[min(binsxnrm) max(binsxnrm) min(binsxnrm) max(binsxnrm)])
xlabel('x_N')
ylabel('y_N')

subplot(3,4,10)
imagesc(1e3*[min(binsxnrm) max(binsxnrm)],1e3*[min(binsynrm) max(binsynrm)],rhoxyf')
set(gca,'YDir','normal')
axis equal
axis(1e3*[min(binsxnrm) max(binsxnrm) min(binsynrm) max(binsynrm)])
xlabel('x_N')
ylabel('y_N')

subplot(3,4,11)
plot(particles(1,nptcles)*1e3,particles(3,nptcles)*1e3,'.k','MarkerSize',1)
axis square
axis([-2 2 -2 2])
psaxis = axis;
xlabel('x (mm)')
ylabel('y (mm)')

subplot(3,4,12)
imagesc(1e3*[pstscale(1,1) pstscale(1,2)],1e3*[pstscale(2,1) pstscale(2,2)],pstxyf')
set(gca,'YDir','normal')
axis equal
axis(psaxis)
xlabel('x (mm)')
ylabel('y (mm)')

set(gcf,'PaperUnits','inches')
set(gcf,'PaperPosition',[0 0 8 6])
print('-dpng','Tomography2D.png')
print('-dpdf','Tomography2D.pdf')