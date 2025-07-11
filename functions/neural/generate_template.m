function template = generate_template(chn_data, template_mode, num_pulse, window_size)
% GENERATE_TEMPLATE
% Creates pulse-by-pulse artifact templates from aligned channel data.

[NSTIM, interpulse_len] = size(chn_data);
template = zeros(size(chn_data));

switch lower(template_mode)
    case 'local'
        for i = 1:NSTIM
            a = floor((i - 1) / num_pulse);
            idx = (1 + num_pulse * a):(min(num_pulse * (a + 1), NSTIM));
            local = movmean(chn_data(idx, :), window_size, 1);
            base = local(i - num_pulse * a, :);
            template(i, :) = base - base(1);
        end
    case 'local_drift_corr'
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
        first_mask = mod((1:NSTIM) - 1, num_pulse) == 0;
        X_first = chn_data(first_mask, :);
        X_nonfirst = chn_data(~first_mask, :);

        k = pca_k;
        if size(X_first, 1) >= k
            [coeff_f, score_f, ~, ~, ~, mu_f] = pca(X_first);
            recon_f = score_f(:, 1:k) * coeff_f(:, 1:k)' + mu_f;
        else
            recon_f = zeros(sum(first_mask), interpulse_len);
        end

        if size(X_nonfirst, 1) >= k
            [coeff_n, score_n, ~, ~, ~, mu_n] = pca(X_nonfirst);
            recon_n = score_n(:, 1:k) * coeff_n(:, 1:k)' + mu_n;
        else
            recon_n = zeros(sum(~first_mask), interpulse_len);
        end

        f_count = 1; nf_count = 1;
        for i = 1:NSTIM
            if first_mask(i)
                temp_i = recon_f(f_count, :); f_count = f_count + 1;
            else
                temp_i = recon_n(nf_count, :); nf_count = nf_count + 1;
            end
            temp_i = temp_i - temp_i(1);
            drift = w * (-mean(temp_i(end-4:end)));
            template(i, :) = temp_i + drift;
        end

    otherwise
        error('Invalid template_mode.');
end
end
