function SE(imgFile, depFile, salFile)
%% SE(imgFile, depFile, salFile)
% Input:
%    imgFile - filename of input rgb image
%    depFile - filename of input depth image
%    salFile - output saliency map 


% ===============parameters===============
segNum = 500; % nubmer of super-pixels
% Single-layer Cellular Automata parameters
theta = 10;  % control the strength of similarity between neighbors
a=0.6;
b=0.2;     % a and b control the strength of coherence

% ===============load input images===============
img = imread(imgFile);
dep = imread(depFile);
if size(img,1)~=size(dep,1) || size(img,2)~=size(dep,2)
    error('RGB image and depth image should have the same size.');
end
if size(img,3) ~= 3
    error('RGB image should have 3 color channels.');
end
if size(dep,3) == 3
    dep = rgb2gray(dep);
end

[h,w,~] = size(img);


% ===============super-pixel segmentation===============
spIdxMap = SLIC_rgbd(img, dep, segNum); % super-pixel label map
spNum = max(spIdxMap(:)); % the actual number of super-pixels
pixelList = cell(spNum, 1);
for ii = 1 : spNum
    pixelList{ii} = find(spIdxMap == ii);
end

img = im2double(img);
dep = im2double(dep);

% ===============Depth-based saliency===============
locValues = [repmat((0:h-1)',[w,1])/(h-1), reshape(repmat(0:w-1,[h,1]),[h*w,1])/(w-1)];
spLoc = zeros(spNum,2);
for ii = 1 : spNum
    spLoc(ii,:) = mean( locValues(spIdxMap==ii,:) );
end
spDepth = zeros(spNum,1);
numDirection = 8;
angStep = 360 / numDirection;
scanLength = 0.4 * sqrt(h^2 + w^2);
for ii = 1 : spNum
    for ang = 0 : angStep : 360
        spDepth(ii) = spDepth(ii) + MinDepDiff(dep, spLoc(ii,2), spLoc(ii,1), ang, scanLength, 5);
    end
end
spDepth = MatNorm(spDepth);

% ===============Color-based saliency===============
adjcMatrix = GetAdjMatrix(spIdxMap, spNum);

meanRgbCol = GetMeanColor(img, pixelList);
meanLabCol = colorspace('Lab<-', double(meanRgbCol));
meanPos = GetNormedMeanPos(pixelList, h, w);
bdIds = GetBndPatchIds(spIdxMap);
colDistM = GetDistanceMatrix(meanLabCol);
posDistM = GetDistanceMatrix(meanPos);
[clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM);

[bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
wCtr = CalWeightedContrast(colDistM, posDistM, bgProb);
spColor = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, wCtr);

% ===============Fusion===============
spSal1 = spDepth .* spColor;
spSal1 = MatNorm(spSal1);
stage1 = zeros(h,w);
for ii = 1 : spNum
    stage1(spIdxMap==ii) = spSal1(ii);
end
stage1 = stage1 .* DepRefine(dep, stage1);
stage1 = MatNorm(stage1);

% ===============CA===============
meanDep = zeros(spNum,1);
for ii = 1 : spNum
    meanDep(ii) = mean(dep(spIdxMap==ii));
end

adjcMatrix = LinkBoundarySPs(adjcMatrix, bdIds);
edges = GetUgraphEdges(adjcMatrix);

colDist = sqrt(sum((meanLabCol(edges(:,1),:) - meanLabCol(edges(:,2),:)).^2,2));
colDist = MatNorm(colDist);
depDist = abs(meanDep(edges(:,1)) - meanDep(edges(:,2)));
depDist = MatNorm(depDist);
weights = exp(-theta*(colDist+depDist));
F = edges2adjMatrix(edges,weights,spNum);

% calculate a row-normalized impact factor matrix
D = sum(F,2);
D = diag(D);
F_norm = D\F;   % the row-normalized impact factor matrix

% compute Coherence Matrix
C = a * MatNorm(1 ./ max(F')) + b;
C_norm = diag(C);

% compute the saliency of each superpixel in prior maps
S = zeros(spNum, 1);
for ii = 1 : spNum
    S(ii) = mean(stage1(spIdxMap==ii));
end
diff = setdiff(1:spNum, bdIds);

% step1: decrease the saliency value of boundary superpixels
for ii = 1 : 5
    S(bdIds) = S(bdIds) - 0.6;
    negInd = find(S < 0);
    if numel(negInd) > 0
        S(negInd) = 0.001;
    end
    S = C_norm * S + (1-C_norm) .* diag(ones(1,spNum)) * F_norm * S;
    S(diff) = MatNorm(S(diff));
end

% step2: control the ratio of foreground larger than a threshold
for ii = 1 : 5
    S(bdIds) = S(bdIds) - 0.6;
    negInd = find(S < 0);
    if numel(negInd) > 0
        S(negInd) = 0.001;
    end
    most_sal_sup = find(S > 0.93);
    if numel(most_sal_sup) < 0.02*spNum
        sal_diff = setdiff(1:spNum, most_sal_sup);
        S(sal_diff) = MatNorm(S(sal_diff));
    end
    S = C_norm * S + (1-C_norm) .* diag(ones(1,spNum)) * F_norm * S;
    S(diff) = MatNorm(S(diff));
end

% step3: simply update the saliency map according to rules
for ii = 1:10
    S = C_norm * S + (1-C_norm) .* diag(ones(1,spNum)) * F_norm * S;
    S = MatNorm(S);
end

salMap=zeros(h,w);
salMap(:)=S(spIdxMap(:));
imwrite(salMap, salFile);
end

%% ~~~~~~~~~~~~~~~
function minDepDiff = MinDepDiff(dep, x, y, angle, arm, step)

[h,w,~] = size(dep);

if angle > 0 && angle <= 45
    xstep = -step;
    ystep = step * tan(angle/180*pi);
elseif angle<=90 
    xstep = -step * tan((90-angle)/180*pi);
    ystep = step;
elseif angle<=135
    xstep = step * tan((angle-90)/180*pi);
    ystep = step;
elseif angle<=180
    xstep = step;
    ystep = step * tan((180-angle)/180*pi);
elseif angle<=225
    xstep = step;
    ystep = -step * tan((angle-180)/180*pi);
elseif angle<=270
    xstep = step * tan((270-angle)/180*pi);
    ystep = -step;
elseif angle<=315
    xstep = -step * tan((angle-270)/180*pi);
    ystep = -step;
elseif angle<=360
    xstep = -step;
    ystep = -step * tan((360-angle)/180*pi);
else
    error('Invalid scan angle');
end

arm = arm * exp(-((x-0.5)^2 + (y-0.5)^2)/0.1);
x = round(x*(w-1)+1);
y = round(y*(h-1)+1);

px = x;
py = y;
accxstep=0;
accystep=0;
mind = 255;
while px<=w && py<=h && px>=1 && py>=1 && sqrt(accxstep^2+accystep^2)<arm
    pd = dep(py,px);
    if pd<mind
        mind = pd;
    end
    accxstep = accxstep + xstep;
    accystep = accystep + ystep;
    px = x + round(accxstep);
    py = y + round(accystep);
end
dp = dep(y,x);

if dp - mind > 0
    minDepDiff = dp - mind;
else
    minDepDiff = 0;
end
end

%% ~~~~~~~~~~~~~~~
function sal = DepRefine(dep, sal)

[h,w,~] = size(dep);
pxNum = h * w;

thr = 255;
while thr >= 0
    if sum(dep>=thr) / pxNum > 0.5
        break;
    end
    thr = thr - 1;
end

mask = dep < thr;
sal(mask) = sal(mask) .* dep(mask) / thr;
end

%% ~~~~~~~~~~~~~~~
function adjMatrix = GetAdjMatrix(idxImg, spNum)
% Get adjacent matrix of super-pixels
% idxImg is an integer image, values in [1..spNum]

[h, w] = size(idxImg);

%Get edge pixel locations (4-neighbor)
topbotDiff = diff(idxImg, 1, 1) ~= 0;
topEdgeIdx = find( padarray(topbotDiff, [1 0], false, 'post') ); %those pixels on the top of an edge
botEdgeIdx = topEdgeIdx + 1;

leftrightDiff = diff(idxImg, 1, 2) ~= 0;
leftEdgeIdx = find( padarray(leftrightDiff, [0 1], false, 'post') ); %those pixels on the left of an edge
rightEdgeIdx = leftEdgeIdx + h;

%Get adjacent matrix of super-pixels
adjMatrix = zeros(spNum, spNum);
adjMatrix( sub2ind([spNum, spNum], idxImg(topEdgeIdx), idxImg(botEdgeIdx)) ) = 1;
adjMatrix( sub2ind([spNum, spNum], idxImg(leftEdgeIdx), idxImg(rightEdgeIdx)) ) = 1;
adjMatrix = adjMatrix + adjMatrix';
adjMatrix(1:spNum+1:end) = 1;%set diagonal elements to 1
adjMatrix = sparse(adjMatrix);
end

%% ~~~~~~~~~~~~~~~
function meanCol = GetMeanColor(image, pixelList)

[h, w, chn] = size(image);
tmpImg=reshape(image, h*w, chn);

spNum = length(pixelList);
meanCol=zeros(spNum, chn);
for i=1:spNum
    meanCol(i, :)=mean(tmpImg(pixelList{i},:), 1);
end
if chn ==1 %for gray images
    meanCol = repmat(meanCol, [1, 3]);
end
end

%% ~~~~~~~~~~~~~~~
function meanPos = GetNormedMeanPos(pixelList, height, width)
% averaged x(y) coordinates of each superpixel, normalized with respect to
% image dimension
% return N*2 vector, row i is superpixel i's coordinate [y x]

spNum = length(pixelList);
meanPos = zeros(spNum, 2);

for n = 1 : spNum
    [rows, cols] = ind2sub([height, width], pixelList{n});    
    meanPos(n,1) = mean(rows) / height;
    meanPos(n,2) = mean(cols) / width;
end
end

%% ~~~~~~~~~~~~~~~
function bdIds = GetBndPatchIds(idxImg, thickness)
% Get super-pixels on image boundary
% idxImg is an integer image, values in [1..spNum]
% thickness means boundary band width

if nargin < 2
    thickness = 8;
end

bdIds=unique([
    unique( idxImg(1:thickness,:) );
    unique( idxImg(end-thickness+1:end,:) );
    unique( idxImg(:,1:thickness) );
    unique( idxImg(:,end-thickness+1:end) )
    ]);
end

%% ~~~~~~~~~~~~~~~
function distM = GetDistanceMatrix(feature)
% Get pair-wise distance matrix between each rows in feature
% Each row of feature correspond to a sample

spNum = size(feature, 1);
DistM2 = zeros(spNum, spNum);

for n = 1:size(feature, 2)
    DistM2 = DistM2 + ( repmat(feature(:,n), [1, spNum]) - repmat(feature(:,n)', [spNum, 1]) ).^2;
end
distM = sqrt(DistM2);
end

%% ~~~~~~~~~~~~~~~
function [clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM)
% Estimate dynamic paras can slightly improve overall performance, but in
% fact, influence of those paras is very small, you can just use fixed
% paras, and we suggest you to set geoSigma = 7, neiSigma = 10.

[meanMin1, meanTop, meanMin2] = GetMeanMinAndMeanTop(adjcMatrix, colDistM, 0.01);
clipVal = meanMin2;

% Emperically choose adaptive sigma for converting geodesic distance to
% weight
geoSigma = min([10, meanMin1 * 3, meanTop / 10]);
geoSigma = max(geoSigma, 5);

% Emperically choose adaptive sigma for smoothness term in Equa(9) of our
% paper.
neiSigma = min([3 * meanMin1, 0.2 * meanTop, 20]);
end

function [meanMin1, meanTop, meanMin2] = GetMeanMinAndMeanTop(adjcMatrix, colDistM, topRate)
% Do statistics analysis on color distances between neighbor patches

spNum = size(adjcMatrix, 1);

% 1. Min distance analysis (between neighbor patches)
adjcMatrix(1:spNum+1:end) = 0;  %patches do not link with itself for min distance analysis
minDist = zeros(spNum, 1);      %minDist(i) means the min distance from sp_i to its neighbors
for id = 1:spNum
    isNeighbor = adjcMatrix(id,:) > 0;
    minDist(id) = min(colDistM(id, isNeighbor));
end
meanMin1 = mean(minDist);

% 2. Largest distance analysis (this measure can reflect image contrast level)
tmp = sort(colDistM(tril(adjcMatrix, -1) > 0), 'descend');
meanTop = mean(tmp(1:round(topRate * length(tmp))));

% 3. Min distance analysis (between 2 layer neighbors)
adjcMatrix = double( (adjcMatrix * adjcMatrix + adjcMatrix) > 0 );  %Reachability matrix
adjcMatrix(1:spNum+1:end) = 0;
minDist = zeros(spNum, 1);      %minDist(i) means the min distance from sp_i to its neighbors
for id = 1:spNum
    isNeighbor = adjcMatrix(id,:) > 0;
    minDist(id) = min(colDistM(id, isNeighbor));
end
meanMin2 = mean(minDist);

if meanMin2 > meanMin1  %as meanMin2 considered more neighbors, its min value should be no larger than meanMin1
    error('meanMin2 should <= meanMin1');
end
end

%% ~~~~~~~~~~~~~~~
function [bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma)
% Estimate background probability using boundary connectivity

bdCon = BoundaryConnectivity(adjcMatrix, colDistM, bdIds, clipVal, geoSigma, true);

bdConSigma = 1; %sigma for converting bdCon value to background probability
fgProb = exp(-bdCon.^2 / (2 * bdConSigma * bdConSigma)); %Estimate bg probability
bgProb = 1 - fgProb;

bgWeight = bgProb;
% Give a very large weight for very confident bg sps can get slightly
% better saliency maps, you can turn it off.
fixHighBdConSP = true;
highThresh = 3;
if fixHighBdConSP
    bgWeight(bdCon > highThresh) = 1000;
end
end

function [bdCon, Len_bnd, Area] = BoundaryConnectivity(adjcMatrix, weightMatrix, bdIds, clipVal, geo_sigma, link_boundary)
% Compute boundary connecity values for all superpixels

if (nargin < 6)
    link_boundary = true;    
end
if (link_boundary)
    adjcMatrix = LinkBoundarySPs(adjcMatrix, bdIds);
end

adjcMatrix = tril(adjcMatrix, -1);
edgeWeight = weightMatrix(adjcMatrix > 0);
edgeWeight = max(0, edgeWeight - clipVal);

% Cal pair-wise shortest path cost (geodesic distance)
geoDistMatrix = graphallshortestpaths(sparse(adjcMatrix), 'directed', false, 'Weights', edgeWeight);

Wgeo = Dist2WeightMatrix(geoDistMatrix, geo_sigma);
Len_bnd = sum( Wgeo(:, bdIds), 2); %length of perimeters on boundary
Area = sum(Wgeo, 2);    %soft area
bdCon = Len_bnd ./ sqrt(Area);
end

function weightMatrix = Dist2WeightMatrix(distMatrix, distSigma)
% Transform pair-wise distance to pair-wise weight using
% exp(-d^2/(2*sigma^2));

spNum = size(distMatrix, 1);

distMatrix(distMatrix > 3 * distSigma) = Inf;   %cut off > 3 * sigma distances
weightMatrix = exp(-distMatrix.^2 ./ (2 * distSigma * distSigma));

if any(1 ~= weightMatrix(1:spNum+1:end))
    error('Diagonal elements in the weight matrix should be 1');
end
end

function adjcMatrix = LinkBoundarySPs(adjcMatrix, bdIds)

adjcMatrix(bdIds, bdIds) = 1;
end

function adjcMatrix = LinkNNAndBoundary(adjcMatrix, bdIds)
%link 2 layers of neighbor super-pixels and boundary patches

adjcMatrix = (adjcMatrix * adjcMatrix + adjcMatrix) > 0;
adjcMatrix = double(adjcMatrix);

adjcMatrix(bdIds, bdIds) = 1;
end

%% ~~~~~~~~~~~~~~~
function wCtr = CalWeightedContrast(colDistM, posDistM, bgProb)
% Calculate background probability weighted contrast

% Code Author: Wangjiang Zhu
% Email: wangjiang88119@gmail.com
% Date: 3/24/2014

spaSigma = 0.4;     %sigma for spatial weight
posWeight = Dist2WeightMatrix(posDistM, spaSigma);

%bgProb weighted contrast
wCtr = colDistM .* posWeight * bgProb;
wCtr = (wCtr - min(wCtr)) / (max(wCtr) - min(wCtr) + eps);

%post-processing for cleaner fg cue
removeLowVals = true;
if removeLowVals
    thresh = graythresh(wCtr);  %automatic threshold
    wCtr(wCtr < thresh) = 0;
end
end

%% ~~~~~~~~~~~~~~~
function optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, fgWeight)

adjcMatrix_nn = LinkNNAndBoundary(adjcMatrix, bdIds);
colDistM(adjcMatrix_nn == 0) = Inf;
Wn = Dist2WeightMatrix(colDistM, neiSigma);      %smoothness term
mu = 0.1;                                                   %small coefficients for regularization term
W = Wn + adjcMatrix * mu;                                   %add regularization term
D = diag(sum(W));

bgLambda = 5;   %global weight for background term, bgLambda > 1 means we rely more on bg cue than fg cue.
E_bg = diag(bgWeight * bgLambda);       %background term
E_fg = diag(fgWeight);          %foreground term

spNum = length(bgWeight);
optwCtr =(D - W + E_bg + E_fg) \ (E_fg * ones(spNum, 1));
end

%% ~~~~~~~~~~~~~~~
function edgeList = GetUgraphEdges(adjMatrix)
% Get edges of a graph from the adjacent matrix
%
% Input:
%    adjMatrix - adjacent matrix representing an undirected graph
%
% Return:
%    edgeList - [edgeNum x 2] matrix
 
adjMatrix = logical(adjMatrix);
nodeNum = size(adjMatrix,1);
adjMatrix(1:nodeNum+1:end) = 0; %set diagonal elements to 0
adjMatrix = adjMatrix + adjMatrix'; % make sure the matrix is symmetric
edgeNum = sum(sum( adjMatrix == 1 )) / 2;
 
edgeList = zeros(edgeNum,2);
p = 0;
for ii = 1 : nodeNum
    nodes = adjMatrix(ii, 1:ii-1);
    nodeIdx = find(nodes);
    n = length(nodeIdx);
    for jj = 1 : n
        edgeList(p + jj, 1) = ii;
        edgeList(p + jj, 2) = nodeIdx(jj);
    end
    p = p + n;
end
end

%% ~~~~~~~~~~~~~~~
function m_norm = MatNorm(m)
 
m = double(m);
max_v = max(m(:));
min_v = min(m(:));
if max_v ~= min_v
    m_norm = (m - min_v) ./ (max_v - min_v + eps);
else
    m_norm = m;
end
end

%% ~~~~~~~~~~~~~~~
function adjMatrix = edges2adjMatrix(edges, weights, nodeNum)
% Given edges and weights, get the weighted adjacency matrix 
%
% Inputs:    
%    edges - [M x 2] list of M edges
%    weights (optional) - [M x 2] list of M weights (default: ones(M,1))
%    nodeNum (optional) - number of nodes in the graph (default: max(edges(:)))
%
% Outputs:   
%    adjMatrix - Adjacency matrix
 
M = size(edges,1);
if nargin < 3
    nodeNum = max(edges(:));
    if nargin < 2
        weights = ones(M,1);
    end
end
 
adjMatrix = sparse([edges(:,1);edges(:,2)], [edges(:,2);edges(:,1)], ...
                   [weights;weights], ...
                   nodeNum, nodeNum);
end