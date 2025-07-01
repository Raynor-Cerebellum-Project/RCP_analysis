function amplifier_data_copy = artifact_Removal(amplifier_data, stim_data, probe_params, template_params, visualize)
    
    

    fs = 30000;
    STIM_CHANS = find(any(stim_data~=0, 2));
    if ~isempty(STIM_CHANS)
    TRIGDAT = stim_data(STIM_CHANS(1),:)';

    trigs1 = find(diff(TRIGDAT) < 0); % Biphasic pulse, first phase negative.
    trigs2 = find(diff(TRIGDAT) > 0);
    if length(trigs1) > length(trigs2)
        trigs  = trigs1;
    else
        trigs = trigs2;
    end
    trigs = trigs(1:2:end);
    
    NSTIM = length(trigs);

    chan_segments = channel_segmentation(STIM_CHANS, probe_params);
    

    template_params.NSTIM = NSTIM;
    
    amplifier_data_copy = amplifier_data;
    switch visualize 
        case "stim"
            visualize_channels = chan_segments.stim;
        case "neighbor-stim"
            visualize_channels = chan_segments.neighbor_stim;
        case "non-stim"
            visualize_channels = chan_segments.non_stim;
        case "all"
            visualize_channels = 1:128;
        otherwise
            visualize_channels = [];
    end

    

    for chan = [chan_segments.stim, chan_segments.neighbor_stim]
        template_params.isstim = true;
        if ismember(chan, visualize_channels)
            amplifier_data_copy(chan, :) = template_subtraction(amplifier_data(chan, :), trigs, chan, template_params);
        else
            amplifier_data_copy(chan, :) = template_subtraction(amplifier_data(chan, :), trigs, 0, template_params);
        end
    end

    for chan = chan_segments.non_stim
        template_params.isstim = false;
        if ismember(chan, visualize_channels)
            amplifier_data_copy(chan, :) = template_subtraction(amplifier_data(chan, :), trigs, chan, template_params);
        else
            amplifier_data_copy(chan, :) = template_subtraction(amplifier_data(chan, :), trigs, 0, template_params);
        end
    end
    else
        amplifier_data_copy = amplifier_data;
    end
    
end