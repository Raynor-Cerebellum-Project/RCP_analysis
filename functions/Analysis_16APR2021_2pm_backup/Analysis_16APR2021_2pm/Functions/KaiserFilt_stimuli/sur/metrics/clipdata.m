function datatobeclipped=clipdata( choice, Sig_show, M_sig_show, Segmentnumbers, Ts)
%CLIPDATA - Copy selected Sig_show from each segment displayed to the clipboard
%
%

%	(c) Claudio G. Rey - 11:11AM  7/8/93
% 	modified Kathleen E. Cullen 10/21/93


NoofSegments = length( M_sig_show(   :, 1));
Noofdata     = length( Sig_show( 1, :));

dataindex    = 1:Noofdata;

if     strcmp(deblank(choice),'max')==1

    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments 
        datatobeclipped( k, :) = max( Sig_show(M_sig_show(k,1):M_sig_show(k,2),dataindex) ); 
    end            

elseif strcmp(deblank(choice),'min')==1

    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments 
        datatobeclipped( k, :) = min( Sig_show(M_sig_show(k,1):M_sig_show(k,2),dataindex) ); 
    end

elseif strcmp(deblank(choice),'mean')==1

    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments
        datatobeclipped( k, :) = mean( Sig_show(M_sig_show(k,1):M_sig_show(k,2),dataindex) ); 
    end

elseif strcmp(deblank(choice),'first')==1
    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments
        datatobeclipped( k, :) = Sig_show(M_sig_show(:,1),dataindex);
    end
elseif strcmp(deblank(choice),'last')==1

    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments
        datatobeclipped( k, :) = Sig_show(M_sig_show(:,2),dataindex);
    end
elseif strcmp(deblank(choice),'delta')==1
    datatobeclipped = zeros( NoofSegments, Noofdata);
    datatobeclipped  = Sig_show(M_sig_show(:,2),dataindex) - Sig_show(M_sig_show(:,1),dataindex);
    
elseif strcmp(deblank(choice),'duration')==1
    datatobeclipped = zeros( NoofSegments, Noofdata);
    
    datatobeclipped = ((M_sig_show(:,2)-M_sig_show(:,1))*Ts)*ones(1,Noofdata);
    
elseif strcmp(deblank(choice),'range')==1

    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments 
        datatobeclipped( k, :) = range( Sig_show(M_sig_show(k,1):M_sig_show(k,2),dataindex) ); 
    end

elseif strcmp(deblank(choice),'max_abs')==1

    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments
        datatobeclipped( k, :) = max(abs( Sig_show(M_sig_show(k,1):M_sig_show(k,2),dataindex) )); 
    end


elseif strcmp(deblank(choice),'location')==1
    datatobeclipped = zeros( NoofSegments, Noofdata);
    for k = 1:NoofSegments 
        datatobeclipped( k, :)  = ((M_sig_show( Segmentnumbers, 2) + M_sig_show( Segmentnumbers, 1))/2*Ts)*ones(1,Noofdata);
    end
elseif strcmp(deblank(choice),'numbers')==1
    datatobeclipped = zeros( NoofSegments, Noofdata);
    
    datatobeclipped  = Segmentnumbers'*ones(1,Noofdata);
   
end