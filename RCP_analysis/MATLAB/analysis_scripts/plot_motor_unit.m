load('NRR_RW027_008_Cal.mat') % load your file
 
spikes = Data.mua(:);
vel = Data.yaw_vel(:);

pref_seg   = Data.segments.contra;
nonpref_seg = Data.segments.ipsi;

pad = 800;                     % ms before/after peak
t = -pad:pad;                 % time axis

figure

%% ================= PREF =================
ntr = size(pref_seg,1);
all_vel = nan(ntr,length(t));

% ----- velocity panel -----
subplot(2,2,1); hold on

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

title('Preferred velocity')
ylabel('Velocity')
xlim([-pad pad])

% ----- raster panel -----
subplot(2,2,3); hold on

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
title('Preferred raster')
xlim([-pad pad])


%% ================= NONPREF =================
ntr = size(nonpref_seg,1);
all_vel = nan(ntr,length(t));

% ----- velocity panel -----
subplot(2,2,2); hold on

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

title('Non-preferred velocity')
xlim([-pad pad])


% ----- raster panel -----
subplot(2,2,4); hold on

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
title('Non-preferred raster')
xlim([-pad pad])

set(gcf,'Renderer','painters')
print(gcf,'raster_plot','-dsvg','-painters')