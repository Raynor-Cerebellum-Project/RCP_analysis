function output = freq_filter_neural(data, params)
    
    flag = params.flag; % 0 for ehp and 1 for ehv
    ehp_left_3d = data.ehp_left;
    ehp_right_3d = data.ehp_right;
    evp_left_3d = data.evp_left;
    evp_right_3d = data.evp_right;
    ehv_left_3d = data.ehv_left;
    ehv_right_3d = data.ehv_right;
    evv_left_3d = data.evv_left;
    evv_right_3d = data.evv_right;



    ehp_left_3d_filtered = data.ehp_left;
    ehp_right_3d_filtered = data.ehp_right;
    evp_left_3d_filtered = data.evp_left;
    evp_right_3d_filtered = data.evp_right;

    ehv_left_3d_filtered = data.ehv_left;
    ehv_right_3d_filtered = data.ehv_right;
    evv_left_3d_filtered = data.evv_left;
    evv_right_3d_filtered = data.evv_right;

    if flag == 0 % if flag ==0, do frequency filter on position data and then use it to calculate velocity
        for i = 1:size(ehp_left_3d, 1)
            ehp_left_3d_filtered{i} = lowpass(ehp_left_3d{i}, params);
            ehp_right_3d_filtered{i} = lowpass(ehp_right_3d{i}, params);
            evp_left_3d_filtered{i} = lowpass(evp_left_3d{i}, params);
            evp_right_3d_filtered{i} = lowpass(evp_right_3d{i}, params);
        end
        for i = 1:size(evp_right_3d,1)
            ehv_left_3d_filtered{i} = gradient(ehp_left_3d_filtered{i});
            ehv_right_3d_filtered{i} = gradient(ehp_right_3d_filtered{i});
            evv_left_3d_filtered{i} = gradient(evp_left_3d_filtered{i});
            evv_right_3d_filtered{i} = gradient(evp_right_3d_filtered{i});
        end
    else % do frequency filter only on the velocity data
        for i = 1:size(evp_right_3d,1)
            ehv_left_3d_filtered{i} = lowpass(ehv_left_3d{i}, params);
            ehv_right_3d_filtered{i} = lowpass(ehv_right_3d{i}, params);
            evv_left_3d_filtered{i} = lowpass(evv_left_3d{i}, params);
            evv_right_3d_filtered{i} = lowpass(evv_right_3d{i}, params);
        end
    end

    output.ehp_left = ehp_left_3d_filtered;
    output.ehp_right = ehp_right_3d_filtered;
    output.evp_left = evp_left_3d_filtered;
    output.evp_right = evp_right_3d_filtered;

    output.ehv_left = ehv_left_3d_filtered;
    output.ehv_right = ehv_right_3d_filtered;
    output.evv_left = evv_left_3d_filtered;
    output.evv_right = evv_right_3d_filtered;

    if isfield(data, 'list')
        output.list = data.list;
    end
    output.ua = data.ua;
    output.fr = data.fr;
end
function filtered_signal = lowpass(signal, params)
%% bandpass filter parameters
% low_pass_freq = params.fc1; % 250Hz
cutoff = params.fc; % 125 Hz
fs = params.fs; % 1000 Hz sampling frequency
%% filter 
% filtered_signal = signal;
% bessel filter Marion R. et.al. 2013
% [b, a] = besself(6, 2 * pi * low_pass_freq);  % Bessel filter (analog, cutoff at 250 Hz)
% 
% % Convert the analog filter to a digital filter using bilinear transformation
% [bd1, ad1] = impinvar(b, a, fs);  % Convert to digital filter with sampling frequency fs



%51st order finite impulse-response filter with a Hamming window and a cutoff at 125Hz)
filter_order = 51;     % 51st-order FIR filter

%Normalize the cutoff frequency (cutoff frequency / Nyquist frequency)
wn = cutoff / (fs / 2);  % Nyquist frequency is fs/2

%Design the 51st-order FIR filter with a Hamming window
fir_coeffs = fir1(filter_order, wn, 'low', hamming(filter_order + 1));
% 


% Apply the filter to signal

% filtered_signal = filtfilt(bd1, ad1, signal);
filtered_signal = filtfilt(fir_coeffs, 1, signal);

end