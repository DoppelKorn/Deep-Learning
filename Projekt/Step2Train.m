
function [net] = Step2Train()
%% Head - General:
%{
Namen:
    S√∂ren Prieﬂ Mat. Nr: 670266
    Bjarne Richter Mat. Nr: 670273
    Salar Haji Ibrahim Mat. Nr: 670366
    Yazan Darwisheh Mat. Nr: 670693
    Marik Malachewski Mat. Nr: 670298
    Finn-Niklas Rathjen Mat. Nr: 670165
%}
%% Head - Specific
%{
Programname:
    Step2Train.m
Funktionsbeschreibung:
    Diese Funktion trainiert das Netz, nachdem die Bilder Augmentiert sind
Version:
    1.0

Changelog:
    0.1 Inital release
    0.2 add Comments
%}

%% Trainingsdaten laden und unterteilen
imageDStrain = imageDatastore("platesResize", "IncludeSubfolders", true, "LabelSource", "foldernames") % Laden der Bilder in Datastore
[imageDStrain, imageDSvalidate] = splitEachLabel(imageDStrain, 0.7, "randomized");                     % Aufteilen der Bilder in Training und Validierung

%% Augmentation
outputSize = [50 200 3]; % Bildgrˆﬂe festlegen am Ausgang
imageAugmenter = imageDataAugmenter("RandRotation", [-5,5],...        % Rotation
                                    "RandScale", [0.8 1.2], ...       % Scale
                                    "RandXShear", [-15,15],...        % Shear X
                                    "RandYShear", [-5 5], ...         % Shear Y
                                    "RandXTranslation", [-10,10],...  % Translation X
                                    "RandYTranslation", [-5 5]);      % Translation Y
imageDStrainAug = augmentedImageDatastore(outputSize, imageDStrain, 'DataAugmentation',imageAugmenter);       % Anwenden des Augmenter auf die Trainingsdaten
imageDSvalidateAug = augmentedImageDatastore(outputSize, imageDSvalidate, 'DataAugmentation',imageAugmenter); % Anwenden des Augmenter auf die Validationsdaten


%% DeepLearning Netzwerk definieren
layers = [                                      % Einleiten der Layer Definition
    imageInputLayer([50 200 3])                 %
    convolution2dLayer(3,8,'Padding','same')    %
    batchNormalizationLayer                     %
    reluLayer                                   %
    maxPooling2dLayer(2,'Stride',2)             %
    convolution2dLayer(3,16,'Padding','same')   %
    batchNormalizationLayer                     %
    reluLayer                                   %   
    maxPooling2dLayer(2,'Stride',2)             %
    convolution2dLayer(3,32,'Padding','same')   %
    batchNormalizationLayer                     %
    reluLayer                                   %
    fullyConnectedLayer(3)                      %
    softmaxLayer                                %
    classificationLayer];                       %

%% Training

options = trainingOptions('sgdm',...            % Festlegen der Trainingsoption, einleitung
    'MaxEpochs',90, ...                         % Anzahl durchl√§ufe  
    'ValidationData', imageDSvalidateAug,...    % Festlegen der Validation DAta(validationImageDS oder validationImageAugDS)
    'ValidationFrequency',15,...                % Festlegen der Validierungsfrequenz
    'Verbose',false,...                         % gibt die Daten des Trainings aus
    'Plots','training-progress');               % Festlegen welche Fenster ausgegebn werden sollen

net = trainNetwork(imageDStrainAug,layers,options);       % Trainieren des Netzwerks (trainingImageDS oder trainingImageAugDS)
predictedLabels = classify(net, imageDSvalidate);         % Vorbereitung der Berechnung der Genauigkeit
accuracy = mean(predictedLabels == imageDSvalidate.Labels)% Berechnen der Genauigkeit


end
