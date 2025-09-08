%SIGTHR - Threshold popup menu
%
%

% (c) Claudio G. Rey - 8:43AM  7/5/93

   loadthrsh;

   list    = 'Minimum|Maximum|Searchout|Signal|Apply Threshold|Do not Apply|Cancel';
   popcall = 'choice = get(hpoplist,''value''); sigthrgt;';

   poplist;
