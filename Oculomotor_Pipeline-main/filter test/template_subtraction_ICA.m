function amplifier_data_copy = template_subtraction_ICA(amplifier_data, trigs, chan, params)
NSTIM = params.NSTIM;
start = params.start;
isstim = params.isstim; % if it is stim channel, we want to blank the signal during stimulus
period = trigs(2)- trigs(1);
period_avg = params.period_avg; % 1ms
if period_avg <= 0 
   period_avg = period; 
end
prebuffer = params.buffer;
chn_data = zeros(NSTIM, period+prebuffer);
rhs_name = params.name;
skip_n = params.skip_n;
time_diffs = diff(trigs);
repeat_gap_threshold = period*2;
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs)];
num_repeats = numel(repeat_boundaries) - 1;
num_pulse = NSTIM/num_repeats;
numComponents = 2;
temp = start:NSTIM;
temp = temp(or(mod(temp, num_pulse)==0, mod(temp, num_pulse) > skip_n));
for i = 1:NSTIM
    segment = (-prebuffer+1 + trigs(i) ):min(period+ trigs(i), length(amplifier_data) );
    chn_data(i, 1:length(segment)) = amplifier_data(segment); 
end
template1 = chn_data;
X = chn_data(temp, :);
[coeff, score, latent] = pca(X);

% Keep components explaining 99% variance
explainedVariance = cumsum(latent) / sum(latent);
numComponents = find(explainedVariance >0.9, 1);

% Reduce and reconstruct
X_reduced = score(:, 1:end) * coeff(:, 1:end)';
% template = template(1:period_avg+prebuffer, :)';

template1(temp, 1:period_avg+prebuffer) = X_reduced(:,  1:period_avg+prebuffer);
for i = 1:NSTIM
    if ~or(mod(i, num_pulse)==0, mod(i, num_pulse) > skip_n)
%         template(i, 1:period_avg+prebuffer) = template(i+skip_n, 1:period_avg+prebuffer);
         template1(i, 1:period_avg+prebuffer) = template1(i+skip_n, 1:period_avg+prebuffer);
    end
    template1(i, 1:period_avg+prebuffer) = template1(i, 1:period_avg+prebuffer)-template1(i, 1) ; %  
    template1(i, period_avg+prebuffer:end) = linspace(template1(i,period_avg+prebuffer), 0, period-period_avg+1);
end
amplifier_data_copy = amplifier_data;
for i = 1 : NSTIM
    segment = (-prebuffer+1 + trigs(i) ):min(period+ trigs(i), length(amplifier_data) );
    amplifier_data_copy( segment) =  template1(i,1:length(segment)); % amplifier_data_copy( segment) -  
 
end

if isstim 
    mark = "(Stim Channel and Neighboring Channel)";
else
    mark = "(Non Stim Channel)";
end


if chan ~= 0 % if we give a channel number, that would let the function plot the artifact removal of the corresponding channel

figure 

plot(amplifier_data, 'linewidth', 1.5, 'color',[0 1 0 0.5],'DisplayName','Original Data')
hold on
plot(amplifier_data_copy,'linewidth', 0.6, 'color',[0 0 1], 'DisplayName', 'After Template Subtraction')

hold off 
legend
xlabel('sample');
ylabel('amplitude');
title(sprintf('channel %i before and after template subtraction ' + mark, chan));

subfolderName = [rhs_name '_channel_data'];
if ~exist(subfolderName, 'dir')
    mkdir(subfolderName)
end
figurename = sprintf('channel_%i_full_' + mark + '.fig', chan);
savepath = fullfile(pwd, subfolderName, figurename);
savefig(savepath)
close
end