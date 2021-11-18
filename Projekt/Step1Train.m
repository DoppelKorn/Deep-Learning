
function [net] = Step1Train()
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
    Step1Train.m
Funktionsbeschreibung:
    Dieses Programm trainiert das NN mit zuf‰lligen Kennzeichen. Die zuf‰lligen Kennzeichen wurden
    mithilfe von "script1_PlateGenerator" erstellt.
Version:
    1.0

Changelog:
    0.1 Inital release
    0.2 add Comments
%}

%% Trainingsdaten laden und unterteilen
imageDStrain = imageDatastore("platesResize", "IncludeSubfolders", true, "LabelSource", "foldernames") % Laden der Bilder in Datastore
[imageDStrain, imageDSvalidate] = splitEachLabel(imageDStrain, 0.7, "randomized");                     % Aufteilen der Bilder in Training und Validierung

%% DeepLearning Netzwerk definieren
layers = [                                      % Einleiten der Layer Definition
    imageInputLayer([50 200 3])                 % Festsetzen der Eingangsbildgr√∂√üe 50 x 200 x 3
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
    'MaxEpochs',30, ...                         % Anzahl durchl√§ufe  
    'ValidationData', imageDSvalidate,...       % Festlegen der Validation DAta(validationImageDS oder validationImageAugDS)
    'ValidationFrequency',10,...                % Festlegen der Validierungsfrequenz
    'Verbose',true,...                          % gibt die Daten des Trainings aus
    'Plots','training-progress');               % Festlegen welche Fenster ausgegebn werden sollen

net = trainNetwork(imageDStrain,layers,options);          % Trainieren des Netzwerks (trainingImageDS oder trainingImageAugDS)
predictedLabels = classify(net, imageDSvalidate);         % Vorbereitung der Berechnung der Genauigkeit
accuracy = mean(predictedLabels == imageDSvalidate.Labels)% Berechnen der Genauigkeit

end
