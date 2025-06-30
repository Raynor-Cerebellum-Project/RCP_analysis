clc
clear all
close all
[datafolder, folderlist] = readfolder("", "");
outputdir = uigetdir(pwd, 'choose a folder to save the output');
p.prebuffer = 100; %prepulse length ms
p.postbuffer = 150; %postpulse length ms
% p.num_pulse_threshold = 5; %An BR file should have >5 pulses
p.fc1 = 125;
p.fc2 = 75;
p.fs = 1000; 
p.flag = 1;
p.threshs = [45, 0.3, 0.6, 0.3, 0.3];
session_name = "Caesar-session-2";
%%
Trialsetting = readtable("D:\Oculomotor Research\Experiment Summary\Experiment_Setting_Summary.xlsx", "Sheet","CRR_NXPL_STIM_002");;

%%
for i = 1:length(folderlist)
    outputfolder = fullfile(outputdir, folderlist{i});
    mkdir(outputfolder)
    folder_path = fullfile(datafolder, folderlist{i});
    [~, filelist] = readfolder(folder_path, '*_neural.mat');
    for j = 1:length(filelist)
        file_path = fullfile(folder_path, filelist{j});
        parts = strsplit(filelist{j}, '-');
        number = parts{1};
        if (number == "023") ||( number == "049")
            continue
        end
        [~, Processed_Data, ~] = pipeline_neural(file_path, p, session_name);
        Refined_Data = post_process_neural(Processed_Data);
        segs = Refined_Data{1};
        ua = Processed_Data.ua;
        timeframe = Processed_Data.timeframe;
        C = bone(length(ua)*2);
        C=flip(C(1:length(ua),:)); 

        if ~isempty(ua)
            dur = Trialsetting.Duration_ms_(Trialsetting.BR == str2double(number));
            channel_num = Trialsetting.num_stim_chan(Trialsetting.BR == str2double(number));
            freq = Trialsetting.Frequency_Hz_(Trialsetting.BR == str2double(number));
            fig = figure;
            
            tiledlayout(4,1)

            nexttile;
            hold on
            plot(segs.timeframe, segs.ipsi_ehp_avg, 'Color','r','LineWidth', 1.5,'DisplayName', number);
            x_plot = [segs.timeframe, fliplr(segs.timeframe)]; 
            y1_plot = [segs.CI_ipsi_ehp_lower, fliplr(segs.CI_ipsi_ehp_upper)];
            fill(x_plot, y1_plot, 1,'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', '95% CI');%fill the confidence interval with color
            xline(0, '--r', 'DisplayName','Stimulus onset');
            xline(dur, 'k--', 'DisplayName',num2str(dur))
            hold off
            grid off
            box off
            axis([-50 segs.timeframe(end) -30 20]);
            title(sprintf("Average Eye horizontal Position stim with %i channels at %i Hz in %i ms",channel_num, freq, dur),'Fontsize',16);
            xlabel("time (ms)")
            ylabel("Eye Horizontal Position (deg)")
            ax = gca;
            ax.FontSize = 16; 


            nexttile;
            hold on
            plot(segs.timeframe, 1000*segs.ipsi_ehv_avg, 'Color','r','LineWidth', 1.5,'DisplayName', number); 
            x_plot = [segs.timeframe, fliplr(segs.timeframe)]; 
            y3_plot = [segs.CI_ipsi_ehv_lower, fliplr(segs.CI_ipsi_ehv_upper)];
            fill(x_plot, 1000*y3_plot, 1,'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', '95% CI');%fill the confidence interval with color
            xline(0, '--r', 'DisplayName','Stimulus onset');
            xline(dur, 'k--', 'DisplayName',num2str(dur))
            hold off
            grid off
            box off
            axis([-50 segs.timeframe(end) -500 300]);
            title(sprintf("Average Eye horizontal Velocity stim with %i channels at %i Hz in %i ms",channel_num, freq, dur),'Fontsize',16);
            xlabel("time (ms)")
            ylabel("Eye Horizontal Velocity (deg/s)")    
            ax = gca;
            ax.FontSize = 16; 


            nexttile;
            hold on
            for row = 1:length(ua)
                ua_seg = ua{row}; % Extract the binary array
                x = find(ua_seg == 1);    % Get indices of 1s
                yStart = row - 0.45;            % Start position of vertical line
                yEnd = row + 0.45;              % End position of vertical line
    
                % Plot vertical lines at each '1' position
                for k = 1:length(x)
                    plot([timeframe(x(k)) timeframe(x(k))], [yStart yEnd], 'Color', C(row,:), 'LineWidth', 2);
                end
            end
        
            % add shaded region that denotes stim duration
            if ~isempty(dur)
                rectangle('Position', [0, 0, dur, length(ua)+1], 'FaceColor', 'yellow', 'FaceAlpha', 0.2,  'EdgeColor', 'none');
            end
            % Formatting
            xlim([-50 segs.timeframe(end)])
            ylim([0 length(ua)+1]);
            xlabel('time (ms)');
            ylabel('trials');
            yticks(1:length(ua));
            title(sprintf("Raster Plot stim with %i channels at %i Hz in %i ms",channel_num, freq, dur),'Fontsize',16);
            grid off;
            box off;
            hold off;
            ax = gca;
            ax.FontSize = 16; 

            nexttile;
            hold on
            plot(segs.timeframe, segs.fr_avg, 'Color','r','LineWidth', 1.5,'DisplayName', number);
            x_plot = [segs.timeframe, fliplr(segs.timeframe)]; 
            y3_plot = [segs.CI_fr_lower, fliplr(segs.CI_fr_upper)];
            fill(x_plot, y3_plot, 1,'FaceAlpha',0.3, 'EdgeColor','none', 'DisplayName', '95% CI');%fill the confidence interval with color

            xline(0, '--r', 'DisplayName','Stimulus onset');
            xline(dur, 'k--', 'DisplayName',num2str(dur))
            hold off
            box off
            axis([-50 segs.timeframe(end) 0 150]);
            title(sprintf("Average Firing Rate for one Cell stim with %i channels at %i Hz in %i ms",channel_num, freq, dur),'Fontsize',16);
            xlabel("time (ms)")
            ylabel("Unit Firing Rate")   
            ax = gca;
            ax.FontSize = 16; 

            figure_id = [number '.fig'];
            savefig(fig, fullfile(outputfolder, figure_id))
            close(fig)
        end
    end
end