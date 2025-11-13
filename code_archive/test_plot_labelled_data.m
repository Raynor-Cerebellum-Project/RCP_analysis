% Step 1: Load the CSV file
filename = 'CamVideo_006_Cam-0DLC_Resnet50_Bert_reach_pilotAug6shuffle1_snapshot_010.csv';  % <-- Update with your actual file name

% Step 1a: Read header separately (first 3 rows)
fid = fopen(filename);
header1 = strsplit(fgetl(fid), ','); % scorer
header2 = strsplit(fgetl(fid), ','); % bodyparts
header3 = strsplit(fgetl(fid), ','); % coords
fclose(fid);

% Step 1b: Read the rest of the table (data from row 4)
opts = detectImportOptions(filename, 'NumHeaderLines', 3);
data = readtable(filename, opts);

% Step 2: Construct full variable names like 'Logan_Fin_x'
nCols = numel(header1);
varNames = cell(1, nCols);

All_kin = table2array(data(:,2:13));

for i = 1:size(All_kin,2)
    sig = All_kin(:,i);
    [b,a] = butter(4,20/(1000/2),'low'); %low pass at 10
    All_kin_filt(:,i)=filtfilt(b,a,sig);
end

%x dimension
figure;plot(All_kin_filt(:,1),'.m')
hold on
plot(All_kin_filt(:,4), '.b')
plot(All_kin_filt(:,7), '.r')
plot(All_kin_filt(:,10), '.y')

%y-dimension
figure;plot(All_kin_filt(:,2),'.m')
hold on
plot(All_kin_filt(:,5), '.b')
plot(All_kin_filt(:,8), '.y')
plot(All_kin_filt(:,11), '.r')
