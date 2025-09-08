%EDITSTR
%
%

% (c) Claudio G. Rey - 9:16PM  1/3/94


   eval( 'delete( heditstr );', '1;'); clear heditstr
   editcall = [editcall 'if exist( ''heditstr'') == 1; delete(heditstr); clear heditstr editcall str; end; axis(axis);'];
   heditstr(1) = figure('Name', 'Edit Window', 'Position', [ 59   282   839    50], 'NumberTitle','off','Color',[0,.5,.5]);
   heditstr(2) = uicontrol('Style','Edit','String',str,'Position',[5,5,1840,20],'Callback', editcall,'HorizontalAlignment','left');
  
	