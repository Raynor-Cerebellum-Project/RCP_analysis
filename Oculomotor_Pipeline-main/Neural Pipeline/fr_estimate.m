
function    fr=fr_estimate(ua,mode,fc,fs)
    if size(ua,1) == 1
        ua = ua.';
    end
    if strcmp(mode,'Kaiser') || strcmp(mode,'kaiser')
        Rp = 0.1;
        Rs = 60;
        ITL = 20;
        dev = [10^(Rp/20), 10^(-Rs/20)];

        Pass_Band = fc;
        Stop_Band = fc+1;
        fcut = [Pass_Band Stop_Band];
        
        for ua_index = 1:size(ua,2)
            fr_tmp = burst(ua(:,ua_index),fs,fcut,ITL,dev);
            fr(:,ua_index) = fr_tmp(1:size(ua,1));
        end
    elseif strcmp(mode,'Causal') || strcmp(mode,'causal')
        a = fc*9.4352;
        dt = 1/fs;
        T = 0:dt:2;
        k = a^2*T.*exp(-a.*T);
        k(k<0) = 0;
        for ua_index = 1:size(ua,2)
            fr_tmp = conv(ua(:,ua_index),k);
            fr(:,ua_index) = fr_tmp(1:end-length(T)+1);
        
        end

    end

end

