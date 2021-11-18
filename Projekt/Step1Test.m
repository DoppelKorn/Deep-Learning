function [] = Step1Test(net)
%% Head - General:
%{
Namen:
    Sören Pries Mat. Nr: 670266
    Bjarne Richter Mat. Nr: 670273
    Salar Haji Ibrahim Mat. Nr: 670366
    Yazan Darwisheh Mat. Nr: 670693
    Marik Malachewski Mat. Nr: 670298
    Finn-Niklas Rathjen Mat. Nr: 670165
%}
%% Head - Specific
%{
Programname:
    Step1Test.m
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

%% ein Bild entnehmen
[T, info] = read(testImageDS);  % DataStore auslesen
str = cellstr(info.Label)       % Ausgeben der wahren Kategorie
image(T)                        % Bild anzeigen
classify(net, T)                % Ausgeben der Antwort des NN

%% Anwendung des trainierten Netzwerkes: Gegen den Gesamtdatensatz testen
predictedLabels = classify(net, testImageDS);           % Vorbereitung der Genauigkeit
accuracy = mean(predictedLabels == testImageDS.Labels)  % Ausgeben der genauigkeit

end