%% Start
clear;
close all;
clc;

%% Ausführen der Skripte
net = Step3Train(); %trainieren des Netzes
Step3Test(net);     %testen des Netzes