function [v,w,ic] = dynfit( dynaparfilename, data, Ts, Mx, sp, mb);
%DYNFIT - Fit second signal in data from dynamics derived from the first signal.
%
%

%  (c) - Claudio G. Rey - 2:15PM  1/15/94

global vconj_temp flag_channel flag_error flag_dyn flag_ini time est pos dyn_res index_pos 

warning off
   
%
%  Load fit parameters from disk:

   [b, nk, f, o, p, FT, nin, nout, sp] = loaddyna( dynaparfilename, Ts);

%
%  See if the offset is an empty variable meaning that no offset is desired:

   if isempty( o) == 1, choice =1; else, choice = 2; end

%
%  Do fit:

   [b,nk,f,o,p,v,w,ic] = fitsegi( choice, data, Ts, Mx, b, nk, f, o, p, FT, 'v', sp);

%

%%%%%%% This section was added for SlideMN %%%%%%%%

   if (flag_dyn == 1),   %as long as this is true, the 17 iterations are not completed, so keep running the loop
     if (index_pos >= 2),
        minX = dyn_res(1,1);    %find the values for the axis
          if (flag_ini ~= 1),
             maxX = minX + 1.6;
          else
             maxX = minX + 4;
          end;
     end;
       
     if (index_pos >= 2), 
        %plot the new results one at the time
        slide_fig = findobj('name','Slide Optimization');
        if (isempty(slide_fig)),
           figure('name','Slide Optimization','position',[232   243   515   479]);
           axis([minX,maxX,0,1]);
           xlabel('Position Coefficient');
           ylabel('VAF');
           hold on;
           grid on;
           pause(1)
        else
           figure(slide_fig);
           pause(1)
        end;    
     
        point = index_pos - 1;
        plot(dyn_res(point,1),dyn_res(point,2),'.')
        title(['Slide Model, Iteration ', num2str(point), '/17 completed']);
        pause(0.1) 
     end;           
     
     Slidnext;             %this is the function that bypasses the menu
     
   elseif (flag_dyn == 2),    %the 17 iterations are completed, but I must re-run the loop a last time with the optimal pos in order to get the parameters
      slide_fig = findobj('name','Slide Optimization');
      figure(slide_fig);
      pause(1)
      point = index_pos - 1;
      plot(dyn_res(point,1),dyn_res(point,2),'.')
      title(['Slide Model, Iteration ', num2str(point), '/17 completed']);
      pause(0.1) 

      bestposindice = find(dyn_res(:,2) == max(dyn_res(:,2)));
      bestpos = dyn_res(bestposindice,1);
      
      disp(' ')
      disp('****** NOW COMPUTE AGAIN WITH THE BEST POSITION SENSITIVITY ******')
      disp(' ')
      disp(['The optimal position sensitivity is: ',num2str(bestpos)]);
      disp(' ')
      
      pos = bestpos;
      flag_dyn = 1;        %set this flag = 1 so that fitsegi goes to Slideend
      index_pos = 99;      %tells Slideend that we are done and that it can set the dyn_flag = 3
      Slidnext;
           
   elseif (flag_dyn == 3),   %means that all the iterations are completed and we can exit the loop
      savedyna( dynaparfilename, b, nk, f, o, p, FT, nin, nout, sp);   %save the parameters of the best model   
      bestposindice = find(dyn_res(:,2) == max(dyn_res(:,2))); bestpos = dyn_res(bestposindice,1);
      
      disp(['The optimal position sensitivity is: ',num2str(bestpos)]);
      disp(' ')
      disp(['Number of restart needed (maximum 2): ' , int2str(flag_error)]);
      disp(' ')
      disp('****** THE FINAL RESULTS ******')
      disp(' ')
      disp('   Pos.       VAF   ')
      disp(dyn_res)   
      
      slide_fig = findobj('name','Slide Optimization');
      if (slide_fig ~= 0),
         figure(slide_fig);
         plot(dyn_res(bestposindice,1),dyn_res(bestposindice,2),'*r');
         hold off;
      end;
      flag_dyn = 0;
      evalin('base','if (exist(''M_temp'') == 1),M = M_temp;end;replot;')
      disp('Done! Really...')
      evalin('base','clear global flag_dyn M_temp')  %clear these global variables so they do not interfere with the next time you run SlideMN
      disp(' ')
      disp('NOTE: "vtar" now contains the final reconstructed estimate')
      disp(' ')
      evalin('base','plot_mod') %plots the optimal fit in the analysis window
      if (flag_channel ~= 1) & (exist('pconj_temp')),
         evalin('base','pconj = pconj_temp;')
      end;
      %mysound;
  elseif (flag_dyn == 4),
      disp(' ')
      disp('Algorithm aborted because of an error...')
      mysound;    
  elseif (flag_dyn == 33),    %when optimizing slide post saccade
      savedyna( dynaparfilename, b, nk, f, o, p, FT, nin, nout, sp);  %if dyn_flag was not set, i.e. SlideMN was not run, do as before and save the results directly
      SlidePost; 
  else   %  Save fit parameters to disk as before: this one is chosen for the regular dynamic analysis 
      savedyna( dynaparfilename, b, nk, f, o, p, FT, nin, nout, sp);  %if dyn_flag was not set, i.e. SlideMN was not run, do as before and save the results directly
  end;
   
   evalin('base','clear global flag_ini flag_dyn flag_one index_pos pos')  %clear these global variables so they do not interfere with the next time you run SlideMN
   
   warning on