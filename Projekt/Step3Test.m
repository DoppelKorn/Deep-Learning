function [] = Step3Test(net)
%% Head - General:
%{
Namen:
    SÃ¶ren Pries Mat. Nr: 670266
    Bjarne Richter Mat. Nr: 670273
    Salar Haji Ibrahim Mat. Nr: 670366
    Yazan Darwisheh Mat. Nr: 670693
    Marik Malachewski Mat. Nr: 670298
    Finn-Niklas Rathjen Mat. Nr: 670165
%}
%% Head - Specific
%{
Programname:
    Step3Test.m
Funktionsbeschreibung:
    Dieses Programm testet das trainierte Netz, mithilfe der Bilder in "PlatesCuttedFromPicAndLabels"
Version:
    1.0

Changelog:
    0.1 Inital release
    0.2 add Comments
%}
%% Testdaten laden
testImageDS = imageDatastore('PlatesCuttedFromPicAndLabels','IncludeSubfolders',true,'LabelSource','foldernames');  % DataStore anlegen

%% Resize Image mit Augmentation
outputSize = [227 227 3];                                             % Bildgröße festlegen am Ausgang
testImageDSResize = augmentedImageDatastore(outputSize, testImageDS); % Anwenden des Augmenter auf die Testdaten um die gewünschte Größe zu erhalten

%% ein Bild entnehmen
[T, info] = read(testImageDSResize);  % DataStore auslesen
str = cellstr(info.Label);            % Ausgeben der wahren Kategorie
classify(net, T);                     % Ausgeben der Antwort des NN

%% Anwendung des trainierten Netzwerkes: Gegen den Gesamtdatensatz testen
predictedLabels = classify(net, testImageDSResize);           % Vorbereitung der Genauigkeit
accuracyTest = mean(predictedLabels == testImageDS.Labels)    % Ausgeben der genauigkeit

end