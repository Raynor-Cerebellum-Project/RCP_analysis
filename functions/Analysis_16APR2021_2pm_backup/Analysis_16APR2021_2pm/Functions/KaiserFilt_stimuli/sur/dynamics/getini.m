function [x0] = getini(x0)

%GUI to change the initial conditions utilized by dodyn2
%Is called when the checkbox "Ini. Cond." is checked in dynatwo
%

P{1} = 'Bias value?';
P{2} = 'First input?';
P{3} = 'Second input?';
P{4} = 'Third input?';
P{5} = 'Fourth input?';

D{1} = num2str(x0(1));
D{2} = num2str(x0(2));
D{3} = num2str(x0(3));
D{4} = num2str(x0(4));
D{5} = num2str(x0(5));

Answer = inputdlg(P,'Replace, if necessary, the default initial conditions',[1],D);

x0(1,1) = str2num(Answer{1});
x0(1,2) = str2num(Answer{2});
x0(1,3) = str2num(Answer{3});
x0(1,4) = str2num(Answer{4});
x0(1,5) = str2num(Answer{5});