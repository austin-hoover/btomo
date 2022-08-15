
close all
clear all

% Specify the resolution (in pixels) of the phase space reconstruction

psresn      = 49; % 61 about the maximum allowed...

% Load the images

nmax        = 40;

imagearray  = cell(1,nmax);

calibration = 0.0181; % Screen 3 millimeters/pixel

disp('Loading the images...'); pause(0.01)

tic

%--------------------------------------------------------------------------

% sigx = 12;
% sigy = 6;
% 
% [xv, yv] = meshgrid(1:psresn,1:psresn);
% 
% phaseAdvance = zeros(nmax,2);
% 
% for n = 1:2:nmax
%     
%     phaseAdvance(n,:) = [pi*n/nmax, pi*rand];
%     
%     xctr = (psresn+1)/2 + 0*psresn/3 * cos(phaseAdvance(n,1));
%     yctr = (psresn+1)/2 + 0*psresn/6 * cos(phaseAdvance(n,2));
%     
%     x1 = xv' - xctr;
%     y1 = yv' - yctr;
%     
%     x2 = x1.*x1/2/sigx/sigx;
%     y2 = y1.*y1/2/sigy/sigy;
%     
%     imgBeam = 100*exp( -x2-y2 );
%     
%     imagesc(imgBeam'); pause(0.1);
% 
%     imagearray{n} = imgBeam;
% 
%     phaseAdvance(n+1,:) = [pi*rand, pi*n/nmax];
%     
%     xctr = (psresn+1)/2 + 0*psresn/3 * cos(phaseAdvance(n+1,1));
%     yctr = (psresn+1)/2 + 0*psresn/6 * cos(phaseAdvance(n+1,2));
%     
%     x1 = xv' - xctr;
%     y1 = yv' - yctr;
%     
%     x2 = x1.*x1/2/sigx/sigx;
%     y2 = y1.*y1/2/sigy/sigy;
%     
%     imgBeam = 100*exp( -x2-y2 );
%     
%     imagearray{n+1} = imgBeam;
%     
%     imagesc(imgBeam'); pause(0.1);
%     
% end

%--------------------------------------------------------------------------

load PhaseSpaceDensityMeasured

v1 = meshgrid(-24:24,1:psresn^3)';
v2 = reshape(v1,[psresn^4,1])';
v3 = reshape(v1,[psresn^3,psresn])';
v4 = reshape(v1,[psresn^2,psresn^2])';
v5 = reshape(v1,[psresn,psresn^3])';
ptcles = [v2(:)'; v3(:)'; v4(:)'; v5(:)'; rho(:)'];

phaseAdvance = zeros(nmax,2);

nmax       = 40;

for n = 1:2:nmax
    
    disp(n);

    mux = pi*n/nmax;
    muy = pi*rand;
    
    phaseAdvance(n,:) = [mux, muy];
    
    psrot = [ cos(mux) sin(mux) 0  0        0          ;...
             -sin(mux) cos(mux) 0  0        0          ;...
              0                 0  cos(muy) sin(muy) 0 ;...
              0                 0 -sin(muy) cos(muy) 0 ;...
              0                 0  0        0        1 ];
          
    ptcles2 = psrot*ptcles;
    
    imgBeam = zeros(psresn,psresn);
    
    for nx = 1:psresn
        
        xi = find(round(ptcles2(1,:))==nx-(psresn-1)/2);
            
        for ny = 1:psresn
            
            yi = find(round(ptcles2(3,:))==ny-(psresn-1)/2);

            xyi = intersect(xi,yi);
            
            imgBeam(nx,ny) = sum(ptcles2(5,xyi));
            
        end
        
    end
    
    imagearray{n} = imgBeam;
    
    mux = pi*rand;
    muy = pi*n/nmax;
    
    phaseAdvance(n+1,:) = [mux, muy];
    
    psrot = [ cos(mux) sin(mux) 0  0        0          ;...
             -sin(mux) cos(mux) 0  0        0          ;...
              0                 0  cos(muy) sin(muy) 0 ;...
              0                 0 -sin(muy) cos(muy) 0 ;...
              0                 0  0        0        1 ];
          
    ptcles2 = psrot*ptcles;
    
    imgBeam = zeros(psresn,psresn);
    
    for nx = 1:psresn
        
        xi = find(round(ptcles2(1,:))==nx-(psresn-1)/2);
            
        for ny = 1:psresn
            
            yi = find(round(ptcles2(3,:))==ny-(psresn-1)/2);

            xyi = intersect(xi,yi);
            
            imgBeam(nx,ny) = sum(ptcles2(5,xyi));
            
        end
        
    end
    
    imagearray{n+1} = imgBeam;
    
end

save('BeamImages.mat','imagearray','phaseAdvance')

%--------------------------------------------------------------------------

% phaseAdvance = zeros(nmax,2);
% 
% for n = 1:2:nmax
%     
%     phaseAdvance(n,:) = [pi*n/nmax, pi*rand];
%     
%     rhoxy   = permute(rho1,[1,3,2,4]);
%     rhoxy   = sum(sum(rhoxy,4),3);
%     
%     xctr = (psresn+1)/2 + 0*psresn/3 * cos(phaseAdvance(n,1));
%     yctr = (psresn+1)/2 + 0*psresn/6 * cos(phaseAdvance(n,2));
%     
%     x1 = xv' - xctr;
%     y1 = yv' - yctr;
%     
%     x2 = x1.*x1/2/sigx/sigx;
%     y2 = y1.*y1/2/sigy/sigy;
%     
%     imgBeam = 100*exp( -x2-y2 );
%     
%     imagesc(imgBeam'); pause(0.1);
% 
%     imagearray{n} = imgBeam;
% 
%     phaseAdvance(n+1,:) = [pi*rand, pi*n/nmax];
%     
%     xctr = (psresn+1)/2 + 0*psresn/3 * cos(phaseAdvance(n+1,1));
%     yctr = (psresn+1)/2 + 0*psresn/6 * cos(phaseAdvance(n+1,2));
%     
%     x1 = xv' - xctr;
%     y1 = yv' - yctr;
%     
%     x2 = x1.*x1/2/sigx/sigx;
%     y2 = y1.*y1/2/sigy/sigy;
%     
%     imgBeam = 100*exp( -x2-y2 );
%     
%     imagearray{n+1} = imgBeam;
%     
%     imagesc(imgBeam'); pause(0.1);
%     
% end

%--------------------------------------------------------------------------


load BeamImages

toc

% return

% Set up variables for the quadrupole scan

xyprojection = zeros(nmax*psresn^2,1);

dfindx       = zeros(2,nmax*psresn^4);
dfcntr       = 1;

ctroffset    = (psresn+1)/2;

% Scan over the quadrupole settings

disp('Setting up the system of equations...'); pause(0.01)

tic

for n = 1:nmax
    
    % Set the phase advance
    
    mux    = phaseAdvance(n,1); % + pi/8; %opticsset(n,2)*pi/180;
    muy    = phaseAdvance(n,2); % + pi/16; %opticsset(n,4)*pi/180;

    cosmux = cos(mux);
    sinmux = sin(mux);
     
    cosmuy = cos(muy);
    sinmuy = sin(muy);
    
    % Find the indices of the non-zero values of the matrix relating the
    % projected distribution at YAG02 to the phase space distribution at YAG01
    
    for yindx1 = 1:psresn
    
        for pyindx1 = 1:psresn
            
            yindx0  = round(cosmuy*(yindx1 - ctroffset) - sinmuy*(pyindx1 - ctroffset) + ctroffset);
            
            if yindx0>0 && yindx0<=psresn

                pyindx0 = round(sinmuy*(yindx1 - ctroffset) + cosmuy*(pyindx1 - ctroffset) + ctroffset);
                
                if pyindx0>0 && pyindx0<=psresn
            
                    for xindx1 = 1:psresn

                        for pxindx1 = 1:psresn

                            xindx0  = round(cosmux*(xindx1 - ctroffset) - sinmux*(pxindx1 - ctroffset) + ctroffset);

                            if xindx0>0 && xindx0<=psresn

                                pxindx0 = round(sinmux*(xindx1 - ctroffset) + cosmux*(pxindx1 - ctroffset) + ctroffset);

                                if pxindx0>0 && pxindx0<=psresn

                                    dfindx(:,dfcntr) = [(n-1)*psresn^2 + (xindx1-1)*psresn + yindx1; (xindx0-1)*psresn^3 + (pxindx0-1)*psresn^2 + (yindx0-1)*psresn + pyindx0];
                                    dfcntr = dfcntr + 1;
            
                                end

                            end

                        end

                    end
                    
                end
                
            end

        end
    
    end
    
    % Construct the vector of (normalised) image pixels
    
    xyprojection(((n-1)*psresn^2+1):n*psresn^2) = reshape(imagearray{n}',[psresn^2,1]);
    
end

% Finally, construct the (sparse) matrix relating the projected
% distribution at YAG02 to the phase space distribution at YAG01

dfindx = dfindx(:,1:dfcntr-1);
dfull  = sparse(dfindx(1,:),dfindx(2,:),ones(1,dfcntr-1),nmax*psresn^2,psresn^4);

toc

% Find the phase space distribution at YAG01 that best fits the images
% observed on YAG02

disp('Solving the system of equations...'); pause(0.01)

tic

rhovector = lsqr(dfull,xyprojection,1e-3,400);

rho       = reshape(rhovector,[psresn,psresn,psresn,psresn]);
rho       = permute(rho,[4 3 2 1]);

toc

% save('PhaseSpaceDensity.mat','rho');



save('PhaseSpaceDensity.mat',...
    'rho',...
    'imgrange',...
    'calibn1',...
    'Beta_x_y_at_reconstruction_point',...
    'Alpha_x_y_at_reconstruction_point',...
    'BeamMomentum');

MakePlots
