vert = gvpos(M+lat);
vert_amp = vert(:,2)-vert(:,1);

hora = ghpos(M+lat);
hora_amp = hora(:,2)-hora(:,1);

vect_amp = sqrt((hora_amp.*hora_amp) + (vert_amp.*vert_amp))
vect_dir = atan ( vert_amp ./ hora_amp)*180/pi

%plot the output on the data display window

Plotlist = str2mat(' fr' , 'ghv' , 'ghp', 'gvp');
NoofDisplayed = length( Plotlist( :, 1));
plotstr = plotmate( Plotlist, Signals, Signaldefinitions);
displaytitle = [dataname ': ']; 
for i = 1:NoofDisplayed, displaytitle = [displaytitle lower(Plotlist(i,1:3)) ' '];end
replot;

top = get(gca,'ylim');
ns_y  = top(2) - (top(2)*.10);
mag_y = top(2) - (top(2)*.15);
dir_y = top(2) - (top(2)*.20);

if timebase == 'tim-',
   x_place = (Md(:,1) + (Mx(:,1) + Mx(:,2))/2);
   for count = 1 : length(ns)
      text(x_place(count)*Ts,mag_y,sprintf('%.1f', vect_amp(ns(count))))
      text(x_place(count)*Ts,dir_y,sprintf('%.1f',vect_dir(ns(count))))
   end
   
else  % timebase == stac
   x_place = (Mx(:,1) + Mx(:,2))/2;
   for count = 1 : length(ns)
      text(x_place(count)*Ts,mag_y,sprintf('%.1f',vect_amp(ns(count))))
      text(x_place(count)*Ts,dir_y,sprintf('%.1f',vect_dir(ns(count))))
   end
   
end

tick = get(gca,'xtick');
left = (tick(1) - tick(2))/1.5;
text(left,ns_y,'ns')
text(left,mag_y,'mag')
text(left,dir_y,'dir')

clear top ns_y mag_y dir_y x_place count tick left


 

  
% Stuff for optimal direction analysis

% Eye position and velocity
% pconj = sqrt((ghpos .* ghpos) + (gvpos .*gvpos))
% vconj = diff (pconj) / 1000;


% vector motor error
% hmerr= -1*((ghpos(1+lat:N))-sh(ghpos(1+lat:N),[M(ns,1),M(ns,2)],+1));
% vmerr= -1*((gvpos(1+lat:N))-sh(gvpos(1+lat:N),[M(ns,1),M(ns,2)],+1));
%vect_err = sqrt((hmerr.*hmerr) + (vmerr.*vmerr));

%vect_err_dir = atan ( vmerr ./ hmerr)*180/pi;

