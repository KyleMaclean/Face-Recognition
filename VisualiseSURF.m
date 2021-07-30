% imgToVisualise = Path to .jpg image

I = processImg(imgToVisualise,100);
points = detectSURFFeatures(I);
imshow(I); hold on;
plot(points.selectStrongest(20));

function img = processImg(filePath,RESOLUTION)
    rgbImg = imread(filePath);
    grayImg = rgb2gray(uint8(rgbImg));
    bbox = step(vision.CascadeObjectDetector('FrontalFaceLBP','MinSize',[200 200]),grayImg);
    if size(bbox,1) == 0
        croppedImg = grayImg;
    else
        croppedImg = imcrop(grayImg, bbox(1,:));
    end
    img = imresize(croppedImg,[RESOLUTION,RESOLUTION]);
end