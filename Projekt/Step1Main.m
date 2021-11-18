%% Start
clear;
close all;
clc;

%% Ausführen der Skripte
net = Step1Train(); %trainieren des Netzes
Step1Test(net);     %testen des Netzes