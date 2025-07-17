function template = generate_template(chn_data, template_mode, repeat_boundaries, window_size, pca_k)
% GENERATE_TEMPLATE
% Creates pulse-by-pulse artifact templates from aligned channel data.


if nargin < 5; pca_k = 3; end                   % Default # PCs


[NSTIM, interpulse_len] = size(chn_data);       % Get size of data
template = zeros(size(chn_data));               % Initialize the template to zeros
w        = linspace(0, 1, interpulse_len);      % Create a vector of time-points to adjust for drift [temporary]

% === Define first-pulse logical index using repeat_boundaries ===
is_first_pulse = false(NSTIM, 1);
is_first_pulse(repeat_boundaries(1:end-1) + 1) = true;

switch lower(template_mode)
    case 'local'
        for i = 1:NSTIM
            if is_first_pulse(i)
                prev_first_pulses = find(is_first_pulse(1:i-1));
                n_available = numel(prev_first_pulses);

                if n_available >= 1
                    n_use = min(3, n_available);
                    ref_idxs = prev_first_pulses(end - n_use + 1:end);
                    base = mean(chn_data(ref_idxs, :), 1);
                    template_i = base - base(1);
                    drift = w * (-template_i(end));
                    template(i, :) = template_i + drift;
                end
                continue
            end

            % Use previous non-first pulses within the window
            count = 0;
            j = i - 1;
            prev_pulses = zeros(window_size, interpulse_len);

            while j > 0 && count < window_size
                if ~is_first_pulse(j)
                    count = count + 1;
                    prev_pulses(count, :) = chn_data(j, :);
                end
                j = j - 1;
            end

            if count > 0
                base = mean(prev_pulses(1:count, :), 1);
            else
                base = zeros(1, interpulse_len);
            end

            template_i = base - base(1);
            drift = w * (-template_i(end));
            template(i, :) = template_i + drift;
        end

    case 'pca'
        for i = 1:NSTIM
        %for i = 1:length(repeat_boundaries)
            if is_first_pulse(i)
                prev_first_pulses = find(is_first_pulse(1:i-1));
                n_available = numel(prev_first_pulses);

                if n_available >= 1
                    n_use = min(3, n_available);
                    ref_idxs = prev_first_pulses(end - n_use + 1:end);
                    base = mean(chn_data(ref_idxs, :), 1);
                    template_i = base - base(1);
                    drift = w * (-template_i(end));
                    template(i, :) = template_i + drift;
                end
             %   continue
            elseif is_first_pulse(i-1)
            % Use last up to 39 previous non-first pulses for PCA
                
                pulse_batch   = ref_idx(i:i+39);
                
                %prev_nonfirst = find(~is_first_pulse(1:i-1));
                %n_available   = numel(prev_nonfirst);
                %if n_available > 0
                %    n_use = min(39, n_available);
                %    ref_idxs = prev_nonfirst(end - n_use + 1:end);
                
                X = chn_data(ref_idxs, :);
                [coeff, score, ~, ~, ~, mu] = pca(X);

                actual_k = min(pca_k, size(score, 2));   % Make sure you're not pulling too many PCs
                recon    = score(end, 1:actual_k)*coeff(:, 1:actual_k)' + mu;
                               
                %else
                %    recon = zeros(1, interpulse_len);  % no data yet
                %end

                template_i = bsxfun(@minus, recon, recon(1,:));
                drift      = bsxfun(@times, w, (-template_i(end,:)));
                template(i:i+39, :) = template_i + drift;
            else
            end
        end
end
end
