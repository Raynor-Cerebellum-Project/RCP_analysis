function [Z_preds, Z_true, rel_mse, corrs] = run_psid_prediction(spike_rates_mat_clean, behavior, nx_size_chosen, n_ahead)

    n1 = 2;  % Output dimensions
    idSys = PSID(spike_rates_mat_clean', behavior, nx_size_chosen, n1, 12);

    A = idSys.A;
    C_y = idSys.Cy;
    C_z = idSys.Cz;

    Y = spike_rates_mat_clean';
    Z = behavior';
    [T, ~] = size(Y);
    valid_T = T - n_ahead;

    Z_preds = nan(valid_T, n_ahead);
    Z_true = nan(valid_T, n_ahead);
    for k = 1:valid_T
        y_k = Y(k, :)';
        x_t = pinv(C_y) * y_k;
        for i = 1:n_ahead
            x_t = A * x_t;
            Z_preds(k, i) = C_z * x_t;
        end
        Z_true(k, :) = Z(k+1:k+n_ahead)';
    end

    rel_mse = sum((Z_preds - Z_true).^2, 1, 'omitnan') ./ sum(Z_true.^2, 1, 'omitnan');

    corrs = nan(1, n_ahead);
    for i = 1:n_ahead
        valid_idx = ~isnan(Z_preds(:, i)) & ~isnan(Z_true(:, i));
        if sum(valid_idx) > 2
            corrs(i) = corr(Z_preds(valid_idx, i), Z_true(valid_idx, i));
        end
    end
end
