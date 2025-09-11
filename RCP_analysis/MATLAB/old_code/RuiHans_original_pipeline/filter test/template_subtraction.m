function amplifier_data_copy = template_subtraction(amplifier_data, trigs, chan, params)
NSTIM = params.NSTIM;
start = params.start;
isstim = params.isstim; % if it is stim channel, we want to blank the signal during stimulus
period = trigs(2)- trigs(1);
period_avg = params.period_avg; % 1ms
prebuffer = params.buffer;
chn_data = zeros(NSTIM, period+prebuffer);
rhs_name = params.name;
skip_n = params.skip_n;
time_diffs = diff(trigs);
repeat_gap_threshold = period*2;
repeat_boundaries = [0; find(time_diffs > repeat_gap_threshold); numel(trigs)];
num_repeats = numel(repeat_boundaries) - 1;
num_pulse = NSTIM/num_repeats;


for i = 1:NSTIM
    segment = (-prebuffer+1 + trigs(i) ):min(period+ trigs(i), length(amplifier_data) );
    chn_data(i, 1:length(segment)) = amplifier_data(segment); 
end
template1 = chn_data;
template2 = chn_data;
temp = start:NSTIM;
temp = temp(or(mod(temp, num_pulse)==0, mod(temp, num_pulse) > skip_n));
temp2 = start:NSTIM;
temp2 = setdiff(temp2, temp);

for i = 1:NSTIM
%     a = floor((i-1)/num_pulse);
%     movemean_idx = (skip_n+1+num_pulse*a):(min(num_pulse*(a+1), NSTIM));
%     template = movmean(chn_data(movemean_idx, 1:period_avg+prebuffer), 5);
    if or(mod(i, num_pulse)==0, mod(i, num_pulse) > skip_n)
%         
%         template1(i, 1:period_avg+prebuffer) = template(i-num_pulse*a-skip_n, :)- template(i-num_pulse*a-skip_n, 1);%  
        template2(i, 1:period_avg+prebuffer) = mean(chn_data(temp, 1:period_avg+prebuffer), 1);
         % 
    else

%     % 
% %         template = mean(chn_data(temp2, 1:period_avg+prebuffer));
%         template1(i, 1:period_avg+prebuffer) = template(1, :); % - template(1, 1)
%         % template2(i, 1:period_avg+prebuffer) = chn_data(i, 1:period_avg+prebuffer);
        template2(i, 1:period_avg+prebuffer) = mean(chn_data(temp2, 1:period_avg+prebuffer), 1);
%        template2(i, 1:period_avg+prebuffer) = template2(i, 1:period_avg+prebuffer) ; % -template2(i, 1)
     end
    % template1(i, period_avg+prebuffer:end) = linspace(template1(i,period_avg+prebuffer), 0, period-period_avg+1);
    template2(i, 1:period_avg+prebuffer) = template2(i, 1:period_avg+prebuffer)-template2(i, 1) ;
    template2(i, period_avg+prebuffer:end) = linspace(template2(i,period_avg+prebuffer), 0, period-period_avg+1);
end
 
% 
% if isstim
%     template = template1;
% else
    template = template2;
% end
amplifier_data_copy = amplifier_data;
for i = 1 : NSTIM
    segment = (-prebuffer+1 + trigs(i) ):min(period+ trigs(i), length(amplifier_data) );
    amplifier_data_copy( segment) = amplifier_data_copy( segment) - template(i,1:length(segment));
  
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