function template = generate_template(chn_data, template_mode, num_pulse, window_size)
% GENERATE_TEMPLATE
% Creates pulse-by-pulse artifact templates from aligned channel data.

[NSTIM, interpulse_len] = size(chn_data);
template = zeros(size(chn_data));

switch lower(template_mode)
    case 'local'
        first_pulse_indices = find(mod((1:NSTIM) - 1, num_pulse) == 0);
        w = linspace(0, 1, interpulse_len);

        for i = 1:NSTIM
            a = floor((i - 1) / num_pulse);
            relative_idx = i - num_pulse * a;

            if relative_idx == 1
                prev_first_pulses = first_pulse_indices(first_pulse_indices < i);
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

            count = 0;
            j = i - 1;
            prev_pulses = zeros(window_size, interpulse_len);

            while j > 0 && count < window_size
                if mod(j - 1, num_pulse) + 1 ~= 1
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
        w = linspace(0, 1, interpulse_len);
        first_pulse_indices = find(mod((1:NSTIM) - 1, num_pulse) == 0);
        is_first = false(NSTIM, 1);
        is_first(first_pulse_indices) = true;
    
        buffer_size = 40;
        template_buffer = zeros(buffer_size, interpulse_len);  % rolling window
        buffer_count = 0;  % number of valid rows in buffer
    
        for i = 1:NSTIM
            if is_first(i)
                % First pulse logic (same as before)
                prev_first_pulses = first_pulse_indices(first_pulse_indices < i);
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
    
            % === PCA using rolling buffer ===
            if buffer_count >= pca_k
                X_prev = template_buffer(1:buffer_count, :);
                [coeff, score, ~, ~, ~, mu] = pca(X_prev);
                recon = score(end, 1:pca_k) * coeff(:, 1:pca_k)' + mu;
            elseif buffer_count > 0
                recon = mean(template_buffer(1:buffer_count, :), 1);  % fallback
            else
                recon = zeros(1, interpulse_len);
            end
    
            % Align and drift-correct
            template_i = recon - recon(1);
            drift = w * (-template_i(end));
            template(i, :) = template_i + drift;
    
            % === Update rolling buffer ===
            if buffer_count < buffer_size
                buffer_count = buffer_count + 1;
                template_buffer(buffer_count, :) = template_i;
            else
                % FIFO shift: discard first row, append new one
                template_buffer = [template_buffer(2:end, :); template_i];
            end
        end

    otherwise
        error('Invalid template_mode.');
end
end
