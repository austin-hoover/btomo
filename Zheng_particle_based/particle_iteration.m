clear all
it = 10;
init = 1;
for iter = init:it

Beam_I =  100; %mA
Mass = 939.294; %Mass(MeV)
startstep = 0.1;
step = 0.4;
%离散测量截面步长 step of measured profile grid
stepx = startstep + step * (1 - (iter - init) / it);%mm
stepy = startstep + step * (1 - (iter - init) / it); %mm

%重建截面步长 step of reconstruction grid
ystep = startstep + step * (1 - (iter - init) / it);
xstep = startstep + step * (1 - (iter - init) / it);

xrange = ceil(36 / stepx) * stepx;%根据包络大小，选择离散截面的范围 range of the measured profiles
yrange = ceil(36 / stepy) * stepy;
pnum = 6; % 用于重建的截面数 number of profiles used in the reconstruction
% 
% % %这一部分为测量包络离散化得到prodis，离散化一次就可以 load the measured profile
% pro1 = dlmread('projection1gauss100mA.txt', '', 3, 0);
% % % pro1 = pro1.data;
% pro2 = dlmread('projection6gauss100mA.txt', '', 3, 0);
% % % pro2 = pro2.data;
% pro3 = dlmread('projection8gauss100mA.txt', '', 3, 0);
% pro4 = dlmread('projection10gauss100mA.txt', '', 3, 0);
% pro5 = dlmread('projection13gauss100mA.txt', '', 3, 0);
% pro6 = dlmread('projection16gauss100mA.txt', '', 3, 0);
% % % pro7 = dlmread('projection7.txt', '', 3, 0);
% % % pro3 = pro3.data;
% pro = {pro1, pro2, pro3, pro4, pro5, pro6};%, pro7
% % for i = 1:pnum
% %     pro{i}(:, 1) = pro{i}(:, 1) * ( - 1);%由于tracewin定义的
% % end
% save 'pro.mat' pro
load pro.mat

prodis = zeros(round(2 * xrange / stepx), round(2 * yrange / stepy), pnum);%
% Discretize the measured profile.
for j = 1:pnum
    for i =  1: length(pro{j}(:, 1))
        if abs(pro{j}(i, 1)) < xrange && abs(pro{j}(i, 3)) < yrange
            prodis(ceil((pro{j}(i, 1) + xrange) / stepx), ceil((pro{j}(i, 3) + yrange) / stepy), j) = prodis(ceil((pro{j}(i, 1) + xrange) / stepx), ceil((pro{j}(i, 3) + yrange) / stepy), j) + 1;
            if isnan(prodis(ceil((pro{j}(i, 1) + xrange) / stepx), ceil((pro{j}(i, 3) + yrange) / stepy), j))
                sprintf('kk');
            end
        end
    end
end
% 这是因为上面求解的分布是转置的，调整回来 
% [Google translation: The distribution solved above is transposed. Adjust it back.]
prodis = permute(prodis, [2 1 3]); 
% Normalize the projections.
for i = 1:pnum
    prodis(:, :, i) = prodis(:, :, i) / sum(sum(prodis(:, :, i)));
end

% save prodis.mat prodis
% load prodis.mat
% %显示离散后prodis
figure(1)
for i = 1:pnum
    subplot(2, 3, i);
    imagesc( - xrange:stepx:xrange - stepx,  - yrange:stepy:yrange - stepy, prodis(:, :, i));
    xlabel('x [mm]');
    ylabel('y [mm]');
end
% if iter == 1
%     N = 1000000;
% else
load parnum.mat
% end
Q2 = [6.46 	4.97 	3.99 	3.47 	4.32 	5.18   ];   % 改变Q铁k值，截面 1, 6, 8, 10, 13, 16
Q3 = [ - 5.63 	 - 4.53 	 - 2.59 	 - 4.01 	 - 4.42 	 - 4.84  ]; % 改变的另一Q铁k值
iter
elenum = 2; % 总共元件数 [Google translation: total number of elements]
% Simulate the measurements.
for i = 1:pnum
    cmd = ['TraceWin.exe XiPAF - IHDTL - beamline_setting1_3_new.ini  current1 = ', num2str(Beam_I), ' dst_file1 = beam', num2str(iter - 1), '.dst ele[2][2] = ', num2str(Q2(i)), ' ele[4][2] = ', num2str(Q3(i)), ' nbr_part1 = ', num2str(N)];
    system(cmd);
    pdata = tracewinplt2txt(elenum);
    temdata = pdata{1, elenum}';
    % 单位转换 [Google translation: unit conversion]
    temdata(:, [1, 3]) = temdata(:, [1, 3]) * 10;
    temdata(:, [2, 4]) = temdata(:, [2, 4]) * 1000;
    pror{i} = temdata;
end
% load prorsimu.mat
% 读入入口分布 [Google translation: Read entry distribution.]
pdo =  pdata{1, 1}';
pdo(:, [1, 3]) = pdo(:, [1, 3]) * 10;
pdo(:, [2, 4]) = pdo(:, [2, 4]) * 1000;


% 添加编号 
% [Google translation: add number]
for i = 1:pnum
    pror{i}(:, 8) = 1:length(pror{1}(:, 1)); % length(pror{1}(:, 1) is the number of particles in the bunch
end
    pdo(:, 8) = 1:length(pror{1}(:, 1));
% 找出在所有截面均存活的粒子，因此后面处理都只在存活的粒子进行, 跟之前tracewin模拟时用所有粒子进行不同 
% [Google Translation: Find the particles that survive in all sections, so 
% the subsequent processing is only performed on the surviving particles, 
% which is different from the previous tracewin simulation with all 
% particles.]
indexloss = [];
for i = 1:pnum
    inde{i} = find(pror{i}(:, 7) ~= 0);
    indexloss = [indexloss; inde{i}];
end
indexloss = unique(indexloss);
pror{pnum}(indexloss, :) = [];
for i = 1:pnum - 1
    interp = ismember(pror{i}(:, 8), pror{pnum}(:, 8));
    in = find(interp == 0);
    pror{i}(in, :) = [];
end

% 每个粒子在不同截面的比重
% [Google translation: The relative density of each particle in different 
% cross-sections.]
ind = zeros(pnum, length(pror{1}(:, 1)));
for j = 1:pnum
    for k = 1:length(pror{j}(:, 1))
        if abs(pror{j}(k, 1)) < xrange && abs(pror{j}(k, 3)) < yrange
            ind(j, k) = prodis(ceil((pror{j}(k, 3) + yrange) / stepy), ceil((pror{j}(k, 1) + xrange) / stepx), j);
        end
    end
end

% 将粒子在多个测量截面的比重相乘得到总比重
% [Google translation: Sum the relative density of the particles in 
% multiple measurement sections to get the total relative density.]
indtemp = ind(1, :);
for i = 2:pnum
    indtemp = indtemp + ind(i, :);
end

% 有效粒子序号，在存活粒子数组中
% [Google translation: cx = effective particle number in the surviving 
% particle array.]
cx = find(indtemp ~= 0);

% 求有效粒子在各个截面分布
% Google translation: Find the distribution of effective particles in each
% section.]
prodispart = zeros(round(2 * xrange / stepx), round(2 * yrange / stepy), pnum);
for j = 1:pnum
    for i = 1:length(cx)
        % This just adds one to the left-hand side.
        prodispart(ceil((pror{j}(cx(i), 1) + xrange) / stepx), ceil((pror{j}(cx(i), 3) + yrange) / stepy), j) = prodispart(ceil((pror{j}(cx(i), 1) + xrange) / stepx), ceil((pror{j}(cx(i), 3) + yrange) / stepy), j) + 1;
   end
end
% 这是因为上面求解的分布是转置的，调整回来 [Google translation: The distribution solved above is transposed. Adjust it back.]
prodispart = permute(prodispart, [2 1 3]); 
prorec = prodispart;
% 归一化 [Google translation: normalization]
for i = 1:pnum
    prorec(:, :, i) = prorec(:, :, i) / sum(sum(prorec(:, :, i)));
end
% 计算重建界面与测量界面间误差 
% [Google translation: Calculate the error between the reconstruct 
% projections and measured projections.]
for i = 1:pnum
    stde(i) = sqrt(sum(sum((prorec(:, :, i) - prodis(:, :, i)).^2)) / numel(prorec(:, :, i)));
end

% 将比例系数对粒子数进行归一化，得到每个粒子所占比重 
% [Google translate: Normalize the scale factor to the number of particles
% to get the proportion of each particle.]
for i = 1:pnum
    prodis(:, :, i) = prodis(:, :, i) / prodispart(:, :, i);
end
prodis(isnan(prodis)) = 0;
prodis(isinf(prodis)) = 0;
% 在有效粒子中求解系数 
% [Google translation: Solve for coefficients in effective particles.]
ind = zeros(pnum, length(pror{1}(:, 1)));
for j = 1:pnum
    for k = 1:length(cx)
        if abs(pror{j}(cx(k), 1)) < xrange && abs(pror{j}(cx(k), 3)) < yrange
            ind(j, cx(k)) = prodis(ceil((pror{j}(cx(k), 3) + yrange) / stepy), ceil((pror{j}(cx(k), 1) + xrange) / stepx), j);
        else 
            ind(j, cx(k)) = 0;
        end
    end
end
% 将粒子在多个测量截面的比重相乘得到总比重 
% [Google translation: Sum the relative density of the particles in 
% multiple measurement sections to get the total relative density.]
index = ind(1, :);
for i = 2:pnum
    index = index + ind(i, :);
end
% 比重归一化 
% [Google translation: Proportion normalization]
index  = index / sum(index);
% 显示重建粒子在三个测量处分布 
% [Google translation: Shows the distribution of
% reconstructed particles at three measurements.]
figure(2);
for i = 1:pnum
    subplot(2, 3, i);
    imagesc( - xrange:stepx:xrange - stepx,  - yrange:stepy:yrange - stepy, prorec(:, :, i));
    xlabel('x [mm]');
    ylabel('y [mm]');
end

% 待重建粒子数
% [Google translation: Number of particles to be reconstructed.]
num = max(500000, length(cx));
% 选出初始分布中的存活粒子
% [Google translation: Select the surviving particles in the initial
% distribution.]
pdi = pdo;
interp = ismember(pdo(:, 8), pror{pnum}(:, 8));
in = find(interp == 0);
pdo(in, :) = [];
% 根据每个粒子系数重新生成分布
% [Google translation: Regenerate the distribution based on the coefficient
% of each particle.]
pdit = zeros(num, 6);
numtep1 = find(num * index < 1.4);
numtep2 = find(num * index >= 0.5);
numinter1 = intersect(numtep1, numtep2); % 1 particle
numtep3 = find(num * index > 1.4); % > 1 particle
pdit(1:length(numinter1), :) = pdo(numinter1, 1:6);
% numtep = intersect(numtep1, numtep2);
% num = num - length(numtep);
istep = 0;
for i = 1:length(numtep3)
    if i == 1
        start = length(numinter1) + 1;
    else
        start = start + istep;
    end
    % 在原粒子所处网格处，在x, x', y, y'方向均匀添加粒子(0~stepx）
    % [Google translation: At the grid where the original particles are 
    % located, add particles uniformly in the x, x', y, y' directions 
    % (0~stepx).]
    istep = round(index(numtep3(i)) * num);
    x = floor((pdo(numtep3(i), 1)) / xstep);
    y = floor((pdo(numtep3(i), 3)) / ystep);
    xp = floor((pdo(numtep3(i), 2)) / xstep);
    yp = floor((pdo(numtep3(i), 4)) / ystep);
    u = unifrnd(0, xstep, istep, 4);
    pri = repmat([x, xp, y, yp], istep, 1);
    pdit(start:start + istep - 1, 1:4) = pri. * repmat([xstep, xstep, ystep, ystep], istep, 1) + u;

  
end
% 储存每次迭代重建截面与测量截面误差
% [Google translation: Store the reconstructed cross-section and the error
% of the measured cross-section for each iteration.]
std = [iter stde];
save('std.txt', 'std', ' - ascii', ' - append');
 
azero = pdit(:, 1) == 0;
pdit(azero, :) = [];
N = length(pdit(:, 1));
if length(pdi(:, 1)) < N
     pdi = repmat(pdi, ceil(N / length(pdi(:, 1))), 1);
end

% 将单位转换成dst要求
% [Google translation: Convert units to dst requirements.]
pdit(:, [1, 3]) =  pdit(:, [1, 3]) / 10;
pdit(:, [2, 4]) =  pdit(:, [2, 4]) / 1000;
pdit(1:N, 5:6) = pdi(1:N, 5:6);

% 保存每次迭代重建的发射度
% [Google translation: Save the reconstructed emittance for each 
% iteration.]
ex = dlmread('partran1.out', '', 14, 0);
emit = [iter ex(1, [16, 17, 26, 27])];
save('emit.txt', 'emit', ' - ascii', ' - append');

% 生成新的dst文件 [Generate a new input particle distribution for the next
% iteration.]
 txt2tracewindst(pdit, iter, Beam_I, Mass);
if iter ~= it
    save parnum.mat N 
    clear 
    it = 10;
    init = 1;
end

end


