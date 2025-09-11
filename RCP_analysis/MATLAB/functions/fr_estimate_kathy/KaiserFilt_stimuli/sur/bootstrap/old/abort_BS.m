function [varargout] = abort_BS(varargin)
% ABORT_BS Application M-file for abort_BS.fig
%    FIG = ABORT_BS launch abort_BS GUI.
%    ABORT_BS('callback_name', ...) invoke the named callback.

% Last Modified by GUIDE v2.0 30-Nov-2000 16:27:27

if nargin == 0  % LAUNCH GUI

	fig = openfig(mfilename,'reuse');

	% Use system color scheme for figure:
	set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));

	% Generate a structure of handles to pass to callbacks, and store it. 
	handles = guihandles(fig);
	guidata(fig, handles);

	if nargout > 0
		varargout{1} = fig;
	end

elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK

	try
		[abort_flag] = feval(varargin{:}); % FEVAL switchyard
	catch
		disp(lasterr);
	end
    
    evalin('caller',' abort_flag = 999;');
    return
    
end


%| ABOUT CALLBACKS:
% --------------------------------------------------------------------

function [x] = pushbutton1_Callback(h, eventdata, handles, varargin)
% Stub for Callback of the uicontrol handles.pushbutton1.

x = 999;
close(handles.figure1)
return

