function f = err1(x)

global tempsig t f1 f2 f3 f4 f5

f = tempsig-((x(1)+(t*0))+(x(2)*(cos(f1*t)))+(x(3)*(sin(f1*t)))+(x(4)*(cos(f2*t)))+(x(5)*(sin(f2*t)))+(x(6)*(cos(f3*t)))+(x(7)*(sin(f3*t)))+(x(8)*(cos(f4*t)))+(x(9)*(sin(f4*t)))+(x(10)*(cos(f5*t)))+(x(11)*(sin(f5*t))));
