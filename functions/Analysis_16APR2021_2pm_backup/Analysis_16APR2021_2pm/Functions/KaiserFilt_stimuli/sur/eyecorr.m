function corr_gaze = eyecorr(data_gaze, data_head)

%
% Function to correct for errors in Bullfrog's __RIGHT_EYE__ 
% position signal caused by head position.
%
% WARNING!!!: This function only corrects the errors in Bullfrog's
% right eye. This function should not be applied to anyone other
% than Bullfrog and it should not be applied to the left eye
% position signal.
%
% Usage:	corr_gaze = reyecorr(control, d_gaze, d_gead)
%
% Where:
%	control = The name of a matlab -MAT file that contains
%		  right gaze position (Ch 0), head position (Ch 2),
%		  and target position (Ch 4) during at least one round
% 		  of the 'sac' paradigm.
%
%	d_gaze  = The gaze position trace that needs to be corrected.
%
%	d_head  = The head position trace that corresponds to d_gaze.
%
% Note: This function is offered 'as is' without any guarantees, 
%       expressed or implied.
%
% GAW: 11/25/97
%
%
%

%function corr_gaze = eyecorr(control, data_gaze, data_head)

%eval(['load ' control]);
%TGP=TGP*20/20.666;

% Get gaze(RE) and head(LE) offsets from calibration file.
%  x = 0; 
%  e0  = cull(mark(TGP(:,1),x-.5,x+.5),100);
%  e0(:,1) = e0(:,1)+500;
%  e0 = join(e0);
%  RE_off = mean(REP(e0));
%  LE_off = mean(LEP(e0)):
  LE_off = 0;
  RE_off = 0;
 
%
  input_len = length(data_gaze);
  corr_gaze = zeros(input_len,1);
  if length(data_head)==1;
    data_head = data_head * ones(input_len,1);
  end

% Subtract offsets from data
  data_head = data_head - LE_off;

% Define correction curve
  pm = [ 0.4093  0.0116];	% For Head < 0
  pp = [ 0.5177 -0.0187];	% For Head > 0
  pg = [-0.0040  0.9800];
% Find points 
  m = find(data_head <  0);
  p = find(data_head >= 0);
  g = find(data_head < -5);
  
% correct gaze
  if ~isempty(m)		% For Head < 0
    corr_gaze(m) = data_gaze(m)- polyval(pm,data_head(m))-RE_off;
  end

  if ~isempty(p)		% For Head > 0
    corr_gaze(p) = data_gaze(p) - polyval(pp,data_head(p))-RE_off;
  end

  if ~isempty(g)
    corr_gaze(g) = corr_gaze(g)./polyval(pg,data_head(g));
  end
%
