function [net] = Step3Train()
%% Head - General:
%{
Namen:
    SÃ¶ren Prieß Mat. Nr: 670266
    Bjarne Richter Mat. Nr: 670273
    Salar Haji Ibrahim Mat. Nr: 670366
    Yazan Darwisheh Mat. Nr: 670693
    Marik Malachewski Mat. Nr: 670298
    Finn-Niklas Rathjen Mat. Nr: 670165
%}
%% Head - Specific
%{
Programname:
    Step3Train.m
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
outputSize = [227 227 3]; % Bildgröße festlegen am Ausgang %50 200 3
imageAugmenter = imageDataAugmenter("RandRotation", [-5,5],...          % Rotation
                                    "RandScale", [0.8 1.2], ...         % Scale
                                    "RandXShear", [-15,15],...          % Shear X
                                    "RandYShear", [-5 5], ...           % Shear Y
                                    "RandXTranslation", [-10,10],...    % Translation X
                                    "RandYTranslation", [-5 5]);        % Translation Y
imageDStrainAug = augmentedImageDatastore(outputSize, imageDStrain, 'DataAugmentation',imageAugmenter);       % Anwenden des Augmenter auf die Trainingsdaten
imageDSvalidateAug = augmentedImageDatastore(outputSize, imageDSvalidate, 'DataAugmentation',imageAugmenter); % Anwenden des Augmenter auf die Validationsdaten


%% DeepLearning Netzwerk definieren
net = alexnet; %Definition des Netztes als Alexnet

layersTransfer = net.Layers(1:end-3)                % Nimmt alle Layer außer die letzten drei (end-3)
numClasses = numel(categories(imageDStrain.Labels)) % Anzahl der Klassen im DS

layers = [                                          %festlegen des Layers   
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, ...
        'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer];

%% Training
options = trainingOptions('sgdm',...            % Festlegen der Trainingsoption, einleitung
    'MiniBatchSize', 10, ...                    % Verlust bewerten und Gewichte aktualisieren
    'MaxEpochs',30, ...                         % Anzahl durchlÃ¤ufe  
    'ValidationData', imageDSvalidateAug,...    % Festlegen der Validation DAta(validationImageDS oder validationImageAugDS)
    'ValidationFrequency',10,...                % Festlegen der Validierungsfrequenz
    'ValidationPatience', 5,...                 % Vorgabe für Abbruch des Trainings
    'InitialLearnRate', 1e-4, ...               % Anfangs Lernrate
    'Verbose',false,...                         % gibt die Daten des Trainings aus
    'Plots','training-progress');               % Festlegen welche Fenster ausgegebn werden sollen

net = trainNetwork(imageDStrainAug,layers,options);          % Trainieren des Netzwerks (trainingImageDS oder trainingImageAugDS)
predictedLabels = classify(net, imageDSvalidateAug);         % Vorbereitung der Berechnung der Genauigkeit
accuracy = mean(predictedLabels == imageDSvalidate.Labels)   % Berechnen der Genauigkeit
end
