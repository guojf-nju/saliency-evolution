function demo()

if ~exist(['SLIC_rgbd.' mexext], 'file')
    mex SLIC_rgbd.cpp;
end

imgDir = './img/';
depDir = './dep/';

resultDir = './result/';
if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end

files = dir([imgDir '*.jpg']);
fileNum = length(files);

msgLength = 0;
for ii = 1 : fileNum
    [~,filename,~] = fileparts(files(ii).name);
    imgFile = [imgDir filename '.jpg'];
    depFile = [depDir filename  '.jpg'];
    
    msgLength = fprintf('Saliency Evolutclcion (%i/%i)\n', ii,fileNum);
    
    SE(imgFile, depFile, [resultDir filename '.png']);
end
