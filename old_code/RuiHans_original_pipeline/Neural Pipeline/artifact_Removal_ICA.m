function amplifier_data_copy = artifact_Removal_ICA(amplifier_data, stim_data, probe_params, template_params, visualize)
    
    

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

    

    template_params.NSTIM = NSTIM;
    
    amplifier_data_copy = amplifier_data;
   

    segments = [];
    for i = 1:NSTIM
        segment = (-prebuffer+1 + trigs(i) ):min(period+ trigs(i), length(amplifier_data) );
        segments = [segments, segment];
    end

    
    
    else
        amplifier_data_copy = amplifier_data; 
    end
    
end