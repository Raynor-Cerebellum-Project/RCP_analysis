%% visualization
function visualize(origin, processed, timeframe, params, filename, session_name, flag)
    description = str2double(regexp(filename, '\d+', 'match'));
    number = description(1);
    density = description(2);
    dur = description(3); % stimulation duration in ms
    freq = description(4); % stimulation frequency in Hz 
    current = description(5); % stimulation current in uA
    n = size(origin.ehp_left,1);
    z = size(processed, 2);
    r = 2*(z+1);
    filters = params.filters;
    if flag == 0
        origin_H_left = origin.ehp_left;
        origin_H_right = origin.ehp_right;
        origin_V_left = origin.evp_left;
        origin_V_right = origin.evp_right;
        title1 = "EHP";
        title2 = "EVP";
        yunit ="(deg)";
        a = 1;
    else
        origin_H_left = origin.ehv_left;
        origin_H_right = origin.ehv_right;
        origin_V_left = origin.evv_left;
        origin_V_right = origin.evv_right;
        title1 = "EHV";
        title2 = "EVV";
        yunit = "(deg/s)";
        a = 1000; % the coefficient to transform the unit

    end

    figure 
    for i = 1:n
        subplot(r,2,1)
        plot(timeframe, a*origin_H_left{i}, 'DisplayName',['stim number ' num2str(i)])
        hold on
    
        subplot(r,2,2)
        plot(timeframe, a*origin_H_right{i}, 'DisplayName',['stim number ' num2str(i)])
        hold on

        subplot(r, 2, z*2+3)
        plot(timeframe, a*origin_V_left{i}, 'DisplayName',['stim number ' num2str(i)])
        hold on

        subplot(r, 2, z*2+4)
        plot(timeframe, a*origin_V_right{i}, 'DisplayName',['stim number ' num2str(i)])
        hold on
    end
   
    for j = 1:z
        D = processed{j};
        if ~isempty(D)
            if flag ==0
                D_H_left = D.ehp_left;
                D_H_right = D.ehp_right;
                D_V_left = D.evp_left;
                D_V_right = D.evp_right;
            else
                D_H_left = D.ehv_left;
                D_H_right = D.ehv_right;
                D_V_left = D.evv_left;
                D_V_right = D.evv_right;
            end
            for i = 1: size(D_H_left,1)
            
                subplot(r, 2, j*2+1)
                plot(timeframe, a*D_H_left{i}, 'DisplayName',['stim number ' num2str(i)])
                hold on

                subplot(r,2,2*j+2)
                plot(timeframe, a*D_H_right{i}, 'DisplayName',['stim number ' num2str(i)])
                hold on
         

                subplot(r, 2, j*2+z*2+3)
                plot(timeframe, a*D_V_left{i}, 'DisplayName',['stim number ' num2str(i)])
                hold on

                subplot(r, 2, j*2+z*2+4)
                plot(timeframe, a*D_V_right{i}, 'DisplayName',['stim number ' num2str(i)])
                hold on
            end
        end
    end


    subplot(r,2,1)
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off

    xlabel("time (ms)")
    ylabel(title1 + " ipsi " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title1 + ' Ipsi before filter', description(1:4)))
    
    subplot(r,2,2)
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off

    xlabel("time (ms)")
    ylabel(title1 +  " contra " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title1 + ' Contra before filter', description(1:4)))
    

    subplot(r, 2, z*2+3)
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off

    xlabel("time (ms)")
    ylabel(title2 +  " ipsi " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title2 + ' Ipsi before filter', description(1:4)))
    

    subplot(r, 2, z*2+4)
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off

    xlabel("time (ms)")
    ylabel(title2 + " contra " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title2 + ' contra before filter', description(1:4)))
    
    for j = 1:z-1
        subplot(r, 2, j*2+1)
        xline(0, 'r--', 'DisplayName',"stimulus onset")
        xline(dur, 'k--', 'DisplayName',"stimulus end")
        hold off
        xlabel("time (ms)")
        ylabel(title1 +  " ipsi " + yunit)
        title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title1 + ' Ipsi after ' + filters{j}, description(1:4)))    

        subplot(r,2,2*j+2)
        xline(0, 'r--', 'DisplayName',"stimulus onset")
        xline(dur, 'k--', 'DisplayName',"stimulus end")
        hold off
        xlabel("time (ms)")
        ylabel(title1 + " contra " + yunit)
        title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title1 + ' Contra after ' + filters{j}, description(1:4)))    
 
         

        subplot(r, 2, j*2+z*2+3)
        xline(0, 'r--', 'DisplayName',"stimulus onset")
        xline(dur, 'k--', 'DisplayName',"stimulus end")
        hold off

        xlabel("time (ms)")
        ylabel(title2 +  " ipsi " + yunit)
        title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title2 + ' Ipsi after ' + filters{j}, description(1:4)))    
 

        subplot(r, 2, j*2+z*2+4)
        xline(0, 'r--', 'DisplayName',"stimulus onset")
        xline(dur, 'k--', 'DisplayName',"stimulus end")
        hold off

        xlabel("time (ms)")
        ylabel(title2 + " contra " + yunit)
        title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title2 + ' Contra after ' + filters{j}, description(1:4)))    
    
    end

    subplot(r, 2, z*2+1)
    hold on
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off
    xlabel("time (ms)")
    ylabel(title1 +  " ipsi " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title1 + ' Ipsi removed by artifact filter', description(1:4)))    

    subplot(r,2,2*z+2)
    hold on
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off
    xlabel("time (ms)")
    ylabel(title1 +  " contra " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title1 + ' Contra removed by artifact filter', description(1:4)))    
 
         

    subplot(r, 2, r*2-1)
    hold on
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off
    xlabel("time (ms)")
    ylabel(title2 +  " ipsi " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title2 + ' Ipsi removed by artifact filter', description(1:4)))    
 

    subplot(r, 2, r*2)
    hold on
    xline(0, 'r--', 'DisplayName',"stimulus onset")
    xline(dur, 'k--', 'DisplayName',"stimulus end")
    hold off
    xlabel("time (ms)")
    ylabel(title2 + " contra " + yunit)
    title(sprintf(session_name + '-%i, with %i channels in %i ms at %i Hz, ' + title2 + ' Contra removed by artifact filter', description(1:4)))    
        

    figurename = erase(filename, '.mat') + sprintf('_'+filters{1} + '_first_%i-fc', params.fc2) + "_" + title1 + "_" + title2+".fig";
    subfolderName = sprintf(session_name + '-fc-%i-Hz',params.fc2);
    if ~exist(subfolderName, 'dir')
        mkdir(subfolderName)
    end
    savepath = fullfile(pwd, subfolderName, figurename);
    savefig(savepath)
end
