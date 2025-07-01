% This is a generic figure "zoom" utility, that can quickly zoom and pan
% across a plotted signal vector, even when it has hundreds of millions of
% data points.

classdef ZoomPlot < handle
	
	properties (SetAccess=public)
		
		% A simple rectangular numeric array/matrix of samples to display.
		% May have multiple columns for multiple channels.
		Values  % Row/Column sense is: Values(NSamps, NChans)
		NCHANS  % This is the number of columns in the Values matrix.
		
		% Start time, and sample interval.
		TIME_start_sec
		SampleInterval_sec
		
		% Data sample index of the leftmost sample currently plotted.
		cursamp=1;
		
		% Number of data samples to display across the plot.
		Nsamps=15000;
		
		% The MinMax struct array, for efficient plotting.
		MM
		
		FIGNUM = 101;
		FIG
		AXES
		LINE      % Array of Line objects, one per data channel.
		LINE_enabled  % Can enable/disable channel display.
		GRID_txbox % TextBox to inform user of grid spacing.
		
		Offset=0; % I guess this should be an array also??
		Scale
		yspan_max
		yspan_min
		
		LastSampNum=0;
		
		MOD_keys={};
		
		%  Boolean to manually turn on markers on the plot trace.
		DOTS_ON = false;
		
		Mouse % Save Mouse info during mouse event callbacks.
		
		% Figure title string.
		FIG_Title_str = ''
		
		
		STATUS_txbox  % For debugging, status, utility, whatever.
	end
	
	% =================================================================
	% STATIC methods.
	methods (Static)
		
	end
	
	% =================================================================
	methods
		% Destructor
		function delete(obj)
			fprintf('%s Destructor called!\n', class(obj));
			if isgraphics(obj.FIG)
				obj.FIG.UserData = [];
				delete(obj.FIG);
			end
		end

		% Constructor. Pass in a struct with .Values and .Times members.
		function obj=ZoomPlot(DAT, fignum)
			if nargin >= 2
				obj.FIGNUM = fignum;
			end
			obj.Values = DAT;
			obj.NCHANS = size(DAT,2);
			obj.MM = obj.MakeMinMax_ALLCHANS(DAT);
			
			data_maxy = max(max(DAT));
			data_miny = min(min(DAT));
			data_yrange = data_maxy - data_miny;
			if data_yrange == 0
				data_yrange = 1;
			end
			
			% See if figure(101) already exists, and keep its position for
			% the new window, and "maximize" it, if it was previously.
			if ishandle(obj.FIGNUM)
				F = figure(obj.FIGNUM);
				F.Units = 'normalized';
				Fkeep = F.Position;     % Remember current window position & size.
				Fstate = F.WindowState; % See if window was maximized.
				close(obj.FIGNUM);             % Kill old window and object.
				
				% This is a little picky. We do a "two step" here. We first
				% position the window, so it winds up on the correct
				% screen, in case there are multiple screens. THEN, if it
				% was full screen, we set the size smaller, then make it
				% full-screen again. That way, when it is taken off of full
				% screen, it has a smaller size than the whole screen.
				F = figure(obj.FIGNUM);     % Open new window.
				F.Units = 'normalized';
				F.Position = Fkeep;  % Restore position & size (puts on correct screen).
				if strcmp(Fstate, 'maximized')   % If it was maximized, do some extra work.
					F.OuterPosition = [Fkeep(1:2) 0.7 0.6];
					F.WindowState = 'maximized';
				end
				F.Units = 'pixels';
				obj.FIG = F;
			else
				obj.FIG=figure(obj.FIGNUM);
				set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.4 0.9 0.6], 'units', 'pixels');
			end
			
			% Create an array of Line objects for the data channels.
			obj.LINE = line(1:obj.Nsamps, zeros(obj.NCHANS,obj.Nsamps), 'linewidth', 0.8);
			obj.LINE_enabled(1:length(obj.LINE)) = true;
			
			%obj.TIME_start_sec = 0;
			%obj.SampleInterval_sec = 1/200000;
			obj.TIME_start_sec = 1;
			obj.SampleInterval_sec = 1;
			
			% Set axes Y range to 10% more than min and max values.
			ylim([data_miny-data_yrange/10 data_maxy+data_yrange/10]);			
			xlim([1 obj.Nsamps]);
			
			grid on;
			obj.AXES = gca;
			obj.Offset = 0;
			%obj.Scale = 1;
			
			obj.GRID_txbox=annotation('textbox', [0 0 0.2 0], 'string', '', 'verticalalignment', 'bottom', 'LineStyle', 'none');
			obj.STATUS_txbox=annotation('textbox', [0.2 0 0.6 0], 'string', '', 'verticalalignment', 'bottom', 'LineStyle', 'none');
			
			% Keep track of mouse info during mouse button presses, and mouse
			% movement.
			obj.Mouse = struct;
			obj.Mouse.Button = [];
			obj.Mouse.StartPos = [0 0];
			obj.Mouse.StartIdx = 1;
			obj.Mouse.StartYLim = obj.AXES.YLim;
			obj.Mouse.OrigPointer = getptr(obj.FIG);
			
			obj.yspan_max = data_maxy + data_yrange/2;
			obj.yspan_min = data_miny - data_yrange/2;
			obj.Scale = 0.94 / 6.475e6 / 2 / 16 * 0.51;
			
			obj.DisplayData();
			
			% Show file name in figure window bar.
			%obj.FIG_Title_str = sprintf('FILE:  %s %s', obj.DAT.fdate, obj.DAT.fname);
			%obj.FIG.Name = obj.FIG_Title_str;
			
			% Add all the callbacks.
			
			obj.FIG.WindowScrollWheelFcn = @obj.ScrollWheelCallBack;
			obj.FIG.KeyPressFcn = @obj.KeyCallback;
			obj.FIG.KeyReleaseFcn = @obj.KeyReleaseCallback;
			obj.FIG.SizeChangedFcn = @obj.SizeChangedCallback;
			
			% Mouse button and movement handlers.
			obj.FIG.WindowButtonDownFcn = @obj.MouseButtonCallback;
			obj.FIG.WindowButtonUpFcn = @obj.MouseButtonUpCallback;
			obj.FIG.WindowButtonMotionFcn = @obj.MouseMoveCallback;
			%obj.FIG.ButtonDownFcn = @obj.MouseButtonDownCallback;
			%obj.AXES.ButtonDownFcn = @obj.MouseButtonDownCallback;
			
			% Delete this object if the figure window is closed.
			obj.FIG.DeleteFcn = @(~,~)delete(obj);
			
			% Call the figure resize function to get things set up.
			obj.SizeChangedCallback(obj.FIG, []);
			
			% Save the current object into Figure(101)'s UserData area.
			obj.FIG.UserData.CurObject = obj;
		end
		
		% Make sure the axes are packed fairly tightly to the outside edges
		% of the figure window, so we don't have vast areas of wasted space
		% outside the axes (as is the typical, default Matlab behavior).
		function SizeChangedCallback(obj, source_obj, evt)
			% Get the total figure window size in pixels.
			fpos = obj.FIG.Position;
			hsize = fpos(3)-70;
			vsize = fpos(4)-70;
			setpixelposition(obj.AXES, [40 50 hsize vsize]);
		end
		
		% How many screen pixels for each data point?
		function ppd = PixPerDatum(obj)
			AxPos_pix = getpixelposition(obj.AXES);
			ppd = AxPos_pix(3) / obj.Nsamps;
			%fprintf('Pix per datum: %.1f\n', ppd);
		end
		
		% Create the rows of Min/Max values for a single channel, to more
		% efficiently display large amounts of zoomed-out data.
		function MM=MakeMinMax(obj, dat)
			InitGroupSize = 20;
			SmallestNC = 4000;
			
			NR = InitGroupSize;          % Number of rows.
			
			MMidx = 0;
			MMdat = dat;
			MM=struct();
			while true
				% Compute #columns, and truncate any extras that are not an
				% exact multiple.
				NC = floor(length(MMdat(:))/NR);  % Number of columns.
				if NC < SmallestNC; break; end
				N = NR * NC;
	
				% Reshape, and take min/max. The output MM array will be
				% 2xNC, so each column of 2 values is the min and max for a
				% group. You may then take MM(:) to get an array of
				% alternating min and max values.
				RS = reshape(MMdat(1:N), NR, []);
				MMdat = [min(RS); max(RS)];
				MMidx = MMidx+1;
				MM(MMidx).Values = MMdat(:);
				MM(MMidx).GroupSize = floor(length(dat) / length(MMdat(:)));
				NR = 4;
				fprintf('-- %2d: %2d x %9d\n', MMidx, NR, NC);
			end
		end
		
		% Loop over all channels to create min/max tables.
		function MM=MakeMinMax_ALLCHANS(obj, DAT)
			fprintf('Computing Min/Max value tables...\n');
			for i=1:size(DAT,2)
				MM(:,i) = obj.MakeMinMax(DAT(:,i));
			end
		end
		
		% Return X-axis times for a given range of analog sample numbers.
		function TIMES = GetTimes(obj, idx_range)
			TIMES = (idx_range-1) * obj.SampleInterval_sec + obj.TIME_start_sec;
		end
		
		% Extract the data from the Min/Max arrays, choosing a row that
		% keeps the number of plotted points to a reasonable value.
		function [T,YData,ROW] = MMGetData(obj, Chan, Sidx, Eidx)
			N_raw = Eidx - Sidx;
			ROW = 0;
			T=[];
			YData=[];
			
			N = N_raw;
			while N > 30000
				NextRow = ROW+1;
				if NextRow > size(obj.MM,1); break; end
				ROW = NextRow;
				N = N_raw / obj.MM(ROW, Chan).GroupSize;
			end
						
			if ROW == 0
				IDX = Sidx:Eidx;
				T = obj.GetTimes(IDX);
				YData = obj.Values(IDX,Chan);
			else
				GS = obj.MM(ROW, Chan).GroupSize;
				NDAT = length(obj.MM(ROW,Chan).Values);
				IDX = max(1,floor(Sidx/GS)):min(NDAT, floor(Eidx/GS));
				T = obj.GetTimes(IDX) * GS;
				YData = obj.MM(ROW, Chan).Values(IDX);
			end			
		end
		
		% Line up data on figure, based on requested zoom.
		function DisplayData(obj)
			% If Nsamps has changed, we make sure that the point currently
			% under the cursor remains "still" on the screen.
			N = obj.Nsamps;
			cur = obj.cursamp;
			DatPerPix = 1/obj.PixPerDatum();
			
			Start_idx = round(max(1,cur));
			End_idx = round(min(length(obj.Values),cur+N-1));
			
			for i=1:length(obj.LINE)
				[T,YData,ROW] = obj.MMGetData(i, Start_idx, End_idx);
				if obj.LINE_enabled(i)
					obj.LINE(i).YData = YData- obj.Offset;
					obj.LINE(i).XData = T;
				else
					obj.LINE(i).XData = [];
					obj.LINE(i).YData = [];
				end
			end
			
			PP = round(getpixelposition(obj.AXES));  % PP(3) is num pixels across axes.
			obj.STATUS_txbox.String = sprintf(...
				'DatPerPix: %.2f  #Points %d  #HPix %d  MM ROW %d', ...
				DatPerPix, max(arrayfun(@(L) length(L.XData), obj.LINE)), ...
				PP(3), ROW);
			
			% The AXES limits are based on on the requested Zoom.
			obj.AXES.XLim = [T(1) T(end)];
			
			% Show dot markers on the trace if we zoom in enough.
			if obj.PixPerDatum() >= 1.0 || obj.DOTS_ON
				[obj.LINE.Marker] = deal('.');
			else
				[obj.LINE.Marker] = deal('none');
			end
			
			% Update string showing user the grid spacing.
			gspace = diff(obj.AXES.XTick);
			gspace = gspace(1);
			%obj.GRID_txbox.String = sprintf('XGrid: %ssec', ScaleString(gspace));
			obj.GRID_txbox.String = sprintf('XGrid: %s samps', ScaleString(gspace));
		end
		
		% Specify zoom as a multiplier of the number of data points. For
		% instance a Zoom of 2 shows twice as much data on the screen (zooms
		% "out"). To keep the dispaly consistent, this should be the ONLY
		% place where obj.Nsamps is modified, since we need to know the
		% "before" and "after" modification to get things right.
		%
		% As we zoom in and out horizontally, we want the data at the CURSOR
		% to be the zoom point, NOT the center of the screen, or the left
		% side of the screen, or any other arbitrary position.
		%
		function ZoomData(obj, Xzoom)
			% Always zoom around the cursor position, so the user can quickly
			% and naturally zoom into a point of interest.
			
			% First, get current "normalized" x cursor pos, where 0 is full
			% left, 1 is full right, and anything in between is a portion of
			% 1.0.
			CP = get(obj.AXES, 'CurrentPoint'); % Mouse current position.
			CPx = CP(1,1);
			CPy = CP(1,2);
			xlims = obj.AXES.XLim;
			spanx = diff(xlims);
			cursor_x_rel = (CPx-xlims(1)) / spanx;
			
			% Convert this cursor position to a data index.
			cursor_idx = round(obj.cursamp + obj.Nsamps*cursor_x_rel);
			
			% The min and max number of samps to display across the figure.
			MIN_SAMPS = 50;
			MAX_SAMPS = min(200000000, length(obj.Values));
			% Now update Nsamps based on requested zoom.
			obj.Nsamps = round(max(MIN_SAMPS, min(MAX_SAMPS, obj.Nsamps * Xzoom)));
			
			% Now update obj.cursamp taking cursor position and new Nsamps
			% into account.
			obj.cursamp = max(1, round(cursor_idx - obj.Nsamps*cursor_x_rel));
		end
		
		% Zoom vertically to the point where the mouse is.
		function ZoomVertical(obj, Yzoom)
			CP = get(obj.AXES, 'CurrentPoint'); % Mouse current position.
			CPy = CP(1,2);
			Ylims = obj.AXES.YLim;
			spanY = diff(Ylims);
			cursor_y_rel = (CPy-Ylims(1)) / spanY;
			
			% Update to the new zoomed span.
			spanY = spanY * Yzoom;
			
			% Now, the lower Y limit should be the span, minus the cursor Y
			% relative position (0.0 to 1.0) times the total span.
			Ylower = CPy - spanY*cursor_y_rel;
			Yupper = Ylower + spanY;
			ylim(obj.AXES, [Ylower Yupper]);
		end
		
		function KeyReleaseCallback(obj, src,event)
			%disp(csprintf('KeyUp, key %s, modifiers: "%s"\n', event.Character, event.Modifier));
			%fprintf('KeyUp %s\n', event.Key);
		end
		
		function KeyCallback(obj, src, event)
			%fprintf('Key %s\n', event.Key);
			
			%if strcmp(event.Key, 'control')
			if strcmp(event.Key, 'shift')
				obj.InitPan();
				return;
			end
			
			switch event.Character
				case '0'
					% Re-center the axes on the plot, taking into account the
					% mean() over the Values of Channel 1.
					MAX = diff(obj.AXES.YLim) / 2;
					%MEAN = mean(obj.Values(1:100:end,1));
					if length(obj.LINE(1).YData) > 0
						MEAN = mean(obj.LINE(1).YData);
					else
						MEAN = mean(obj.LINE(2).YData);
					end
					obj.AXES.YLim = [-MAX MAX] + MEAN;
					
				case {'1','2','3','4','5','6','7','8','9'}
					chan = event.Character - '0';
					if chan <= length(obj.LINE_enabled)
						obj.LINE_enabled(chan) = ~obj.LINE_enabled(chan);
					end
				case 'd'
					obj.DOTS_ON = ~obj.DOTS_ON;

				case 'x'
					% Toggle exponents on and off for X tick labels.
					ax = obj.AXES;
					
					if strcmp(ax.XRuler.ExponentMode, 'auto')
						ax.XRuler.Exponent=0;
						xtickformat('%.0f');
					else
						ax.XRuler.ExponentMode = 'auto';
						xtickformat('auto')
					end
			
			end %switch
			obj.DisplayData();
		end
		
		% Return three Booleans that tell whether any of the SHIFT, CTRL, or
		% ALT keys are currently pressed.
		function [SHIFT, CTRL, ALT] = GetKeyboardMods(obj)
			% NICE, this gets a cell array of strings with zero to three of
			% 'shift', 'control', and 'alt', in any combination! Perfect.
			mod = obj.FIG.CurrentModifier;
			%disp(mod);
			SHIFT = any(strcmp(mod, 'shift'));
			CTRL = any(strcmp(mod, 'control'));
			ALT = any(strcmp(mod, 'alt'));
		end
		
		% Set obj.cursamp, and make sure it stays within limits, and is
		% always an integer.
		function SetCursamp(obj, NewCursamp)
			obj.cursamp = round(max(1, min(length(obj.Values)-obj.Nsamps, NewCursamp)));
		end
		
		function CP = ScrollWheelCallBack(obj, src,evt)
			[SHIFT, CTRL, ALT] = obj.GetKeyboardMods();
			
			CP = get(obj.AXES, 'CurrentPoint'); % Mouse current position.
			
			scrollcnt = evt.VerticalScrollCount;
			%fprintf('Mouse wheel scroll count: %d\n', scrollcnt);
			
			if ALT
				yspan = obj.AXES.YLim(2) - obj.AXES.YLim(1);
				if (scrollcnt < 0) && (yspan > obj.yspan_min)
					%ylim = ylim / 2;
					obj.ZoomVertical(0.5);
				elseif (scrollcnt > 0) && (yspan < obj.yspan_max)
					%ylim = ylim * 2;
					obj.ZoomVertical(2.0);
				end
				
			elseif CTRL && ~SHIFT
				% Zoom horizontally.
				if scrollcnt < 0
					%obj.Nsamps = max(100, obj.Nsamps/2);
					obj.ZoomData(0.5);
				else
					%obj.Nsamps = min(500000, obj.Nsamps*2);
					obj.ZoomData(2.0);
				end
			else
				% Pan horizontally. SHIFT pans by 0.4 screen, no SHIFT pans
				% by 1/20th screen.
				PANsz = round(obj.Nsamps * cond(SHIFT, cond(CTRL, 1, 0.4), 0.05)); % Pan 1/20th of the screen width.
				obj.SetCursamp(obj.cursamp - scrollcnt*PANsz);
			end
			
			obj.DisplayData();
			CP = get(obj.AXES, 'CurrentPoint'); % Mouse current position.
		end
		
		% 		function [L, R, M] = GetMouseKeyStatus(obj)
		% 			if ~ispc
		% 				error('Running under Windows only.');
		% 			end
		% 			if ~libisloaded('user32')
		% 				loadlibrary('C:\WINDOWS\system32\user32.dll', 'user32.h');
		% 			end
		% 			L = calllib('user32', 'GetAsyncKeyState', int32(1)) ~= 0;
		% 			R = calllib('user32', 'GetAsyncKeyState', int32(2)) ~= 0;
		% 			M = calllib('user32', 'GetAsyncKeyState', int32(4)) ~= 0;
		% 		end
		
		%
		function InitPan(obj)
			%obj.Mouse.StartPos = obj.AXES.CurrentPoint;
			% Get mouse position IN PIXELS!! If we get CurrentPoint in X
			% units, things get confused, since we change the XData as a
			% result of the panning motion.
			obj.Mouse.StartPos = obj.FIG.CurrentPoint;
			obj.Mouse.StartIdx = obj.cursamp;
			obj.Mouse.StartYLim = obj.AXES.YLim;
			obj.Mouse.StartOffset = obj.Offset;
			%fprintf('WheelDown: "%s"  CP %.3f, CurSamp %d, ylim [%.1f %.1f]\n', ...
			%   seltype, obj.Mouse.StartPos(1), obj.Mouse.StartIdx, obj.Mouse.StartYLim);
		end
		
		function MouseButtonCallback(obj, src, event)
			% Save the SelectionType here, and then MouseMove can decide (if
			% it is 'extend'==middle-button) whether to act upon the movement.
			seltype = event.Source.SelectionType;
			%disp(seltype);
			% 			[L,R,M] = obj.GetMouseKeyStatus();
			% 			fprintf('Mouse button: "%s"  %s %s %s\n', seltype, L, M, R);
			% 			fprintf('Mouse button: "%s"\n', seltype);
			
			obj.Mouse.Button = seltype;
			obj.InitPan();
			obj.Mouse.OrigPointer = getptr(obj.FIG);
			if strcmp(seltype, 'extend')
				setptr(obj.FIG, 'closedhand');
			end
			
			% Turn off X axis auto scaling while we are moving.
			obj.Mouse.OrigXLimMode = obj.AXES.XLimMode;
			obj.AXES.XLimMode = 'manual';
			obj.AXES.YLimMode = 'manual';
		end
		
		% 		function MouseButtonDownCallback(obj, src, event)
		% 			fprintf('MouseButtonDown  %s\n', event.Source.SelectionType);
		% 			%fprintf('MouseButtonDown  AXES\n');
		% 		end
		
		function MouseButtonUpCallback(obj, src, event)
			seltype = event.Source.SelectionType;
			%fprintf('Mouse button UP: "%s"\n', seltype);
			
			obj.Mouse.Button = [];
			obj.AXES.XLimMode = obj.Mouse.OrigXLimMode;
			obj.AXES.YLimMode = obj.Mouse.OrigXLimMode;
			
			setptr(obj.FIG, 'arrow');
		end
		
		function cp=MouseMoveCallback(obj, src, event)
			cp = obj.AXES.CurrentPoint;
			%obj.FIG.Name = sprintf('Mouse: %.1f %.1f', cp(1), cp(2));
			
			% Hopefully this will not slow us down too much. Show the
			% stimulation block that the mouse is near.
			%
			% Actual usage: This seems very efficient. I imagine it is just
			% setting a string somewhere in memory, and does not try to
			% actually write out to the figure window except when the screen
			% is "refreshed". Does not seem to slow down the cursor at all.
			%         obj.FIG.Name = sprintf('%s     %s', ...
			%            obj.FIG_Title_str, obj.DAT.STM.StimNearTime_str(cp(1)));
			
			% Move the data back and forth with the mouse. The
			% MouseButtonCallback() remembers the initial settings when the
			% mouse button was first pushed, so that we can set the X position
			% based on the mouse delta movement.
			if strcmp(obj.Mouse.Button, 'extend') || ... % Mouse MIDDLE button.
					... %any(strcmp(obj.FIG.CurrentModifier, 'control')) % CTRL key held down
					any(strcmp(obj.FIG.CurrentModifier, 'shift')) % SHIFT key held down
				% First, get current "normalized" x cursor pos, where 0 is full
				% left, 1 is full right, and anything in between is a portion of
				% 1.0.
				% Mouse position IN PIXELS (from Figure, NOT from Axes).
				CP = obj.FIG.CurrentPoint;
				CPx = CP(1);
				CPy = CP(2);
				
				% Amount of cursor horizontal relative motion from the start
				% when the button was first pressed.
				AxPos_pix = getpixelposition(obj.AXES);
				cursor_x_rel = (CPx - obj.Mouse.StartPos(1)) / AxPos_pix(3);
				
				% Now update the start sample position.
				obj.SetCursamp(obj.Mouse.StartIdx - cursor_x_rel*obj.Nsamps);
				
				% Update the vertical offset.
				cursor_y_rel = (CPy - obj.Mouse.StartPos(2)) / AxPos_pix(4);
				
				% Move the axes up and down.
				obj.AXES.YLim = obj.Mouse.StartYLim - cursor_y_rel*diff(obj.Mouse.StartYLim);
				obj.DisplayData();
			end
		end
		
	end %methods
end %classdef



% Some extra functions.

% "Conditional" operator, like C/C#, but need a method call for it. Also,
% both arguments are ALWAYS evaluated, as they must be, in order to be
% passed to this function.
%
% If BOOL is True, return first arg, else second arg.
function val = cond(BOOL, val1, val2)
if BOOL
	val = val1;
else
	val = val2;
end
end
