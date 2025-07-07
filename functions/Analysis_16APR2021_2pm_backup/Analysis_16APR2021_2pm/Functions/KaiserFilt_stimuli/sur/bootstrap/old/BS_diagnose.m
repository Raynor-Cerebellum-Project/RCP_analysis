function [mean_check] = BS_diagnose(DATA,obs,ns)

mean_check = zeros(size(obs,2),size(DATA,2));

for i = 1:size(obs,2),
    idf_t = find(sum(obs' == ns(i)) == 0);
    if ~isempty(idf_t)
        if (size(idf_t,2) > 1),
            mean_check(i,:) = mean(DATA(idf_t,:));
        else
            mean_check(i,:) = (DATA(idf_t,:));
        end      
    else
        mean_check(i,:) = NaN;
    end
end

    