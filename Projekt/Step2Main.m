%% Start
clear;
close all;
clc;

%% Ausführen der Skripte
net = Step2Train(); %trainieren des Netzes
Step2Test(net);     %testen des Netzes