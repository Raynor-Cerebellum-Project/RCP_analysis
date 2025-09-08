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
    num_blocks = length(repeat_boundaries) - 1;

    for b = 1:num_blocks
        % Get pulse indices for this block
        i_start = repeat_boundaries(b) + 1;
        i_end   = repeat_boundaries(b+1);
        pulse_batch = i_start:i_end;

        % === Step 1: First pulse correction ===
        first_pulse_idx = i_start;
        prev_first_pulses = is_first_pulse(1:first_pulse_idx - 1);
        prev_first_idxs = find(prev_first_pulses);

        if ~isempty(prev_first_idxs)
            n_use = min(3, numel(prev_first_idxs));
            ref_idxs = prev_first_idxs(end - n_use + 1:end);
            base = mean(chn_data(ref_idxs, :), 1);
        else
            base = zeros(1, interpulse_len);
        end

        template_i = base - base(1);
        drift      = w * (-template_i(end));
        template(first_pulse_idx, :) = template_i + drift;

        % === Step 2: PCA on rest of block ===
        if i_end > i_start  % only if more than 1 pulse in block
            pca_indices = i_start+1 : i_end;
            X = chn_data(pca_indices, :);

            % Optional: subtract mean per pulse to isolate shape
            X_centered = X - mean(X, 2);

            [coeff, score, ~, ~, ~, mu] = pca(X_centered);
            actual_k = min(pca_k, size(score, 2));

            if actual_k > 0
                recon = score(:, 1:actual_k) * coeff(:, 1:actual_k)' + mu;
            else
                recon = repmat(mu, size(X, 1), 1);
            end

            % Apply endpoint-aligned drift correction
            template_i = recon - recon(:, 1);  % Optional if already centered
            drift = bsxfun(@times, w, -template_i(:, end));
            template(pca_indices, :) = template_i + drift;
        end

    end
end
end
