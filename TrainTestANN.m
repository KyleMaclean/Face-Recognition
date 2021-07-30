function outputLabel = TrainTestANN(trainPath, testPath)

    % importedNetPath = ...
    classesPath = 'CASIA-WebFace-Classes.mat';
    
    load(classesPath,'classes');
    net = importONNXNetwork(importedNetPath,'OutputLayerType','classification','Classes',classes);
    inputSize = net.Layers(1).InputSize;

    imdsTrain = imageDatastore(trainPath,'IncludeSubfolders',true,'LabelSource','foldernames');
    imdsTest = imageDatastore(testPath,'IncludeSubfolders', true,'LabelSource','none');
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

    featureLayer = "drop7";
    featuresTrain = activations(net,augimdsTrain,featureLayer,'OutputAs','rows');
    classifier = fitcecoc(featuresTrain,imdsTrain.Labels,'Coding','onevsall');
    featuresTest = activations(net,augimdsTest,featureLayer,'OutputAs','rows');
    outputLabel = char(predict(classifier,featuresTest));

end