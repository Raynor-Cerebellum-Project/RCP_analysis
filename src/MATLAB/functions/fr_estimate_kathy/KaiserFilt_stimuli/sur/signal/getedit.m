%GETEDIT - Get a redefined signal from the user using a string box at the bottom of the screen
%
%

% Claudio G. Rey - 12:24PM  8/12/93


%  str1 - remove command
%  str2 - string containing signal to be edited.
%  buf1 - pointer subset array for signals to be displayed used by remove
%  buf2 - scratch temporary holder
%  buf3 - scratch temporary holder

hpopdisplayed = []; hgetedit = [];

%  Delete if dialog box already exists:

   eval(' delete( heditsignaldefs);','1;');

%  Initialize:

   Signalno = 1; k = 1;

%  Define calls:

   hpopsignalscall    = 'Signalno = get(hpopsignals,''value''); eval(dohgetedit);';
   hpopdisplayedcall  = 'Signaldisplayed = get(hpopdisplayed,''value''); Signalno = listfind(Plotlist(Signaldisplayed,:),Signals);eval(dohgetedit);';
   hgeteditcall       = 'str2 = get(hgetedit,''String'');[Signaldefinitions] = listedit(Signalno,str2,Signaldefinitions);';
   hgeteditnoshowcall = 'k=get(hpopdisplayed,''value'');if NoofDisplayed>1,buf1=[1:k-1 k+1:NoofDisplayed];Plotlist=Plotlist(buf1,:);NoofDisplayed=NoofDisplayed-1;end; eval(dohpopdisplayed);';
%    hgeteditlastcall   = 'k=get(hpopdisplayed,''value'');buf2=(NoofDisplayed,:);Plotlist(NoofDisplayed,:)=Plotlist(k,:);Plotlist(k,:)=buf2;eval(dohpopdisplayed);';
   hgeteditlastcall   = 'k=get(hpopdisplayed,''value'');buf2=Plotlist(NoofDisplayed,:);Plotlist(NoofDisplayed,:)=Plotlist(k,:);Plotlist(k,:)=buf2;eval(dohpopdisplayed);';
   hgeteditfirstcall  = 'k=get(hpopdisplayed,''value'');buf3=Plotlist(1,:);Plotlist(1,:)=Plotlist(k,:);Plotlist(k,:)=buf3;eval(dohpopdisplayed);';
   hgeteditshowcall   = 'Plotlist = str2mat(Plotlist,Signals(Signalno,:));NoofDisplayed=NoofDisplayed+1;eval(dohpopdisplayed);';
   
%  Define dynamic controls

   dohgetedit         = 'eval(''delete( hgetedit      );'',''1;''); hgetedit      = uicontrol(''Style'',''Edit'',''String'',deblank(Signaldefinitions(Signalno,:)), ''Position'', [   5,   5, 590,  20],''Callback'', hgeteditcall,''horizontalalignment'',''left'');';
   dohpopdisplayed    = 'eval(''delete( hpopdisplayed );'',''1;''); hpopdisplayed = uicontrol(''Style'',''Popup'',''String'',makelist(Plotlist,1,colorlist),        ''Position'', [ 297, 45, 100, 150],''Callback'', hpopdisplayedcall);';

%  Define and paint static controls

   heditsignaldefs = figure(    'name',  'Signal Definitions',                                 'Position', [   23   108   600   200], 'IntegerHandle','off','color',[.25,.75,.75],'menubar','none','nextplot','replace','resize','off');
   hpopsignals     = uicontrol( 'Style', 'Popup',      'String', makelist(Signaldescriptions), 'Position', [   5, 45, 200, 150], 'Callback', hpopsignalscall);
   hgeteditnoshow  = uicontrol( 'Style', 'Pushbutton', 'String',                     'Remove', 'Position', [ 405, 170,  80,  25], 'Callback', hgeteditnoshowcall,'horizontalalignment','left');
   hgeteditfirst   = uicontrol( 'Style', 'Pushbutton', 'String',                      'First', 'Position', [ 405, 120,  80,  25], 'Callback', hgeteditfirstcall,'horizontalalignment','left');
   hgeteditlast    = uicontrol( 'Style', 'Pushbutton', 'String',                       'Last', 'Position', [ 405,  70,  80,  25], 'Callback', hgeteditlastcall,'horizontalalignment','left');
   hgeteditshow    = uicontrol( 'Style', 'Pushbutton', 'String',                     'Add ->', 'Position', [ 210, 170,  80,  25], 'Callback', hgeteditshowcall,'horizontalalignment','left');
   hgeteditok      = uicontrol( 'Style', 'Pushbutton', 'String',                       'Show', 'Position', [ 500,  80,  85, 105], 'Callback', 'plotnew;','horizontalalignment','left');

%  Evaluate and paint dynamic controls

   eval(dohpopdisplayed);
   eval(dohgetedit);
