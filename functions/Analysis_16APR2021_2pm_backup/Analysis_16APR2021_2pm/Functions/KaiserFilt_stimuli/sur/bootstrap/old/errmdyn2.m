function f =errmdyn2(x,t_out,t_in)

f = t_out - ( x(1) + (x(2).* t_in(:,1)) + (x(3).* t_in(:,2)) + (x(4).* t_in(:,3)) + (x(5).* t_in(:,4)) );

