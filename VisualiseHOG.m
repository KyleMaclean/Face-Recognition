bestHyperParameters = [4356 8 100 200];

HOG_FEATURE_LENGTH = bestHyperParameters(1);
HOG_CELL_SIZE = bestHyperParameters(2);
IMG_RESIZE_SCALE = bestHyperParameters(3);
MIN_FACE_SIZE = bestHyperParameters(4);

% imgToVisualise = Path to .jpg image

[hogFeatures,hogVisualisation] = hogPipeline(imgToVisualise,HOG_CELL_SIZE, IMG_RESIZE_SCALE, MIN_FACE_SIZE);
plot(hogVisualisation);

function [hogFeatures,hogVisualisation] = hogPipeline(filePath,HOG_CELL_SIZE, IMG_RESIZE_SCALE, MIN_FACE_SIZE)
    rgbImg = imread(filePath);
    grayImg = rgb2gray(uint8(rgbImg));
    bbox = step(vision.CascadeObjectDetector('FrontalFaceLBP','MinSize',[MIN_FACE_SIZE MIN_FACE_SIZE]),grayImg);
    if size(bbox,1) == 0
        croppedImg = grayImg;
    else
        croppedImg = imcrop(grayImg, bbox(1,:));
    end
    resizedImg = imresize(croppedImg,[IMG_RESIZE_SCALE IMG_RESIZE_SCALE]);
    [hogFeatures,hogVisualisation] = extractHOGFeatures(resizedImg,'CellSize',[HOG_CELL_SIZE HOG_CELL_SIZE]);
end