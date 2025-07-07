% sp.m

% set-up for smooth pursuit analysis

array = []; new_sig = []; nseg = []; dist_last = [];

% -------------
disp('----------------------------------------------------------------')  
disp('Prepare smooth pursuit stitch file for dynamic multiple analysis')
disp('  1) Find saccades in both directions')
disp('  2) Find where pursuit is within criteria')
disp('  3) adjust M to represent the pursuit portions of the data')
disp('  4) run dynatwo.m (PA code)')
disp(' ')

% ------ set pursuit criteria
thres_value = 20;
P{1} = 'Threshold value?';
D{1} = num2str(thres_value);
Answer = inputdlg(P,'Enter pursuit threshold',[1],D);
thres_value = str2num(Answer{1});

%------------ window around saccades 
new_sig = ghv - diff(htar*1000);
array = mark(abs(new_sig), thres_value, 30000, 0);
array = [(array(:,1)-5),(array(:,2)+5)];

%------------  inmiddle
array = sort(array(:));
array = array(2:length(array));
array = [array; array(length(array)) + 200];

array = [array(1:2:length(array)), array(2:2:length(array))];
array = cull(array,20);

% ----------- check that M is not too long
nseg = size(array,1);
dist_last = (size(ghpos,1)-1) - array(nseg,2);

while (dist_last < 50),
    array(nseg,:) = [];
    disp('M was clipped not to exceed length of data')
    nseg = nseg - 1;
    dist_last = (size(ghpos,1)-1) - array(nseg,2);
end;

% ----------- set M
M = array;

% ----------- replot all segments
ns = 1:length(M(:,1));
replot

% ----------- clean up
clear array new_sig nseg dist_last P D thres_value Answer

% -----------  call PA analysis

dynatwo
