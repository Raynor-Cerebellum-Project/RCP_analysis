data_loc = '/Volumes/data/Current Project Databases - NHP/2025 Cerebellum prosthesis/Nike/20260326_NRR_RW027_fastig/Calibrated/IntanFile_9/NRR_RW027_009_Cal.mat';
load(data_loc);
%%
spikes = Data.MU1(:);
vel = Data.yaw_vel(:);

pref_seg   = Data.segments.contra;
nonpref_seg = Data.segments.ipsi;

pad = 800;                     % ms before/after peak
t = -pad:pad;                 % time axis

%%
figure
%% Preferred direction
ntr = size(pref_seg,1);
all_vel = nan(ntr,length(t));

% Velocity panel
subplot(2,2,1); hold on;
set(gca, 'TickDir', 'out');

for tr = 1:ntr
   
    s = pref_seg(tr,1);
    e = pref_seg(tr,2);

    [~,idx] = max(abs(vel(s:e)));
    peak = s + idx - 1;

    s2 = max(1, peak-pad);
    e2 = min(length(vel), peak+pad);

    seg_vel = vel(s2:e2);
    seg_vel_db = seg_vel-mean(seg_vel(5:150));

    all_vel(tr,1:length(seg_vel_db)) = seg_vel_db;

    plot(t(1:length(seg_vel_db)), seg_vel_db,'Color',[.8 .8 .8])
end

plot(t,nanmean(all_vel,1),'r','LineWidth',2)

title('Contra velocity')
ylabel('Velocity')
ylim([-100, 500])
xlim([-pad pad])

% ----- raster panel -----
subplot(2,2,3); hold on;
set(gca, 'TickDir', 'out');

for tr = 1:ntr
   
    s = pref_seg(tr,1);
    e = pref_seg(tr,2);

    [~,idx] = max(abs(vel(s:e)));
    peak = s + idx - 1;

    s2 = max(1, peak-pad);
    e2 = min(length(spikes), peak+pad);

    seg_spikes = spikes(s2:e2);

    spike_times = find(seg_spikes);
    spike_times = (s2:e2);
    spike_times = spike_times(seg_spikes==1) - peak;

    for k = 1:length(spike_times)
        line([spike_times(k) spike_times(k)],[tr-.4 tr+.4],'Color','k')
    end
end

xlabel('Time from peak velocity (ms)')
ylabel('Trial')
title('Contra raster')
xlim([-pad pad])


%% ================= NONPREF =================
ntr = size(nonpref_seg,1);
all_vel = nan(ntr,length(t));

% ----- velocity panel -----
subplot(2,2,2); hold on;
set(gca, 'TickDir', 'out');

for tr = 1:ntr
   
    s = nonpref_seg(tr,1);
    e = nonpref_seg(tr,2);

    [~,idx] = max(abs(vel(s:e)));
    peak = s + idx - 1;

    s2 = max(1, peak-pad);
    e2 = min(length(vel), peak+pad);

    seg_vel = vel(s2:e2);
    seg_vel_db = seg_vel-mean(seg_vel(5:150));

    all_vel(tr,1:length(seg_vel_db)) = seg_vel_db;

    plot(t(1:length(seg_vel_db)), seg_vel_db,'Color',[.8 .8 .8])
end

plot(t,nanmean(all_vel,1),'r','LineWidth',2)

title('Ipsi velocity')
ylim([-400, 100])
xlim([-pad pad])


% ----- raster panel -----
subplot(2,2,4); hold on;
set(gca, 'TickDir', 'out');

for tr = 1:ntr
   
    s = nonpref_seg(tr,1);
    e = nonpref_seg(tr,2);

    [~,idx] = max(abs(vel(s:e)));
    peak = s + idx - 1;

    s2 = max(1, peak-pad);
    e2 = min(length(spikes), peak+pad);

    seg_spikes = spikes(s2:e2);

    spike_times = find(seg_spikes);
    spike_times = (s2:e2);
    spike_times = spike_times(seg_spikes==1) - peak;

    for k = 1:length(spike_times)
        line([spike_times(k) spike_times(k)],[tr-.4 tr+.4],'Color','k')
    end
end

xlabel('Time from peak velocity (ms)')
ylabel('Trial')
title('Ipsi raster')
xlim([-pad pad])

set(gcf,'Renderer','painters')
set(gca, 'TickDir', 'out');
print(gcf,'raster_plot','-dsvg','-painters')