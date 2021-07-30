function outputLabel = TrainTestHOG(trainPath, testPath)

    % Best Hyper Parameters
    HOG_FEATURE_LENGTH = 4356;
    HOG_CELL_SIZE = 8;
    IMG_RESIZE_SCALE = 100;
    MIN_FACE_SIZE = 200;

    folderNames=ls(trainPath);
    trainImages=zeros(IMG_RESIZE_SCALE,IMG_RESIZE_SCALE,length(folderNames)-2);
    allTrainFeatures=zeros(HOG_FEATURE_LENGTH,size(trainImages,3));
    trainLabels=folderNames(3:end,:);
    testImgNames=ls([testPath, '*.jpg']);
    outputLabel = char(zeros([size(testImgNames, 1), 6]));

    for j = 1:length(folderNames)-2
        imgName = ls([trainPath, trainLabels(j, :), '\*.jpg']);
        filePath = [trainPath, trainLabels(j,:), '\', imgName];
        trainFeatures = hogPipeline(filePath,HOG_CELL_SIZE,IMG_RESIZE_SCALE,MIN_FACE_SIZE);
        allTrainFeatures(:,j) = trainFeatures(:);
    end

    faceClassifier = fitcecoc(allTrainFeatures', trainLabels,'Coding','onevsall');

    for i=1:size(testImgNames,1)
        filePath = [testPath, testImgNames(i,:)];
        testFeatures = hogPipeline(filePath,HOG_CELL_SIZE,IMG_RESIZE_SCALE,MIN_FACE_SIZE);
        outputLabel(i,:) = predict(faceClassifier, testFeatures);
    end
    
end

function hogFeatures = hogPipeline(filePath,HOG_CELL_SIZE, IMG_RESIZE_SCALE, MIN_FACE_SIZE)
    
    rgbImg = imread(filePath);
    grayImg = rgb2gray(uint8(rgbImg));
    bbox = step(vision.CascadeObjectDetector('FrontalFaceLBP','MinSize',[MIN_FACE_SIZE MIN_FACE_SIZE]),grayImg);
    if size(bbox,1) == 0
        croppedImg = grayImg;
    else
        croppedImg = imcrop(grayImg, bbox(1,:));
    end
    resizedImg = imresize(croppedImg,[IMG_RESIZE_SCALE IMG_RESIZE_SCALE]);
    hogFeatures = extractHOGFeatures(resizedImg,'CellSize',[HOG_CELL_SIZE HOG_CELL_SIZE]);

end