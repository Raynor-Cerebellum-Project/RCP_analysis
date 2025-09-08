DEFINT A-Z
DIM ins(1 TO 8)
DIM i AS LONG, ix AS LONG, offset AS LONG, indx(1 TO 10000) AS LONG
DIM atype AS LONG, mrows AS LONG, ncols AS LONG
DIM imagef AS LONG, namlen AS LONG
DIM namevar AS STRING * 5, char0 AS STRING * 1


    PRINT "File: ", COMMAND$
  
    name$ = COMMAND$

    k = INSTR(name$, ".") - 1

REM    INPUT "Enter name "; name$
  
    dtname$ = LEFT$(name$, k) + "d" + RIGHT$(name$, 4)
   
    OPEN name$ FOR INPUT AS #1
    OPEN dtname$ FOR BINARY AS #2
   
    INPUT #1, ins(1)
   
    atype = 130
    mrows = 60000
    ncols = 8
    imagf = 0
    namlen = 6
    namevar = "array"
    char0 = STRING$(1, 0)
   

    PUT #2, , atype
    PUT #2, , mrows
    PUT #2, , ncols
    PUT #2, , imagef
    PUT #2, , namlen
    PUT #2, , namevar
    PUT #2, , char0
  
    PRINT "Reading and Writing data array"
   
    FOR j = 1 TO 10

       FOR k = 1 TO 6000

          INPUT #1, ins(1), ins(2), ins(3), ins(4), ins(5), ins(6), ins(7), ins(8)
          PUT #2, , ins(1)
          PUT #2, , ins(2)
          PUT #2, , ins(3)
          PUT #2, , ins(4)
          PUT #2, , ins(5)
          PUT #2, , ins(6)
          PUT #2, , ins(7)
          PUT #2, , ins(8)
         
       NEXT k

       PRINT j, "x 10%"

    NEXT j
   
    PRINT "Reading and discarding data padding"
   
    FOR k = 1 TO 416

       INPUT #1, ins(1), ins(2), ins(3), ins(4), ins(5), ins(6), ins(7), ins(8)
 
    NEXT k
   
    PRINT "Reading index array"

    ix = 0
    offset = 1
    DO UNTIL EOF(1)

       INPUT #1, ins(1)

       IF ins(1) = -32768 THEN
          offset = offset + 32768
       ELSEIF ins(1) > 0 THEN
         
          ix = ix + 1
          indx(ix) = offset + ins(1)
         
       END IF
       
    LOOP

    PRINT "Writing index array"

    atype = 20
    mrows = ix
    ncols = 1
    imagf = 0
    namlen = 6
    namevar = "index"
    char0 = STRING$(1, 0)
  

    PUT #2, , atype
    PUT #2, , mrows
    PUT #2, , ncols
    PUT #2, , imagef
    PUT #2, , namlen
    PUT #2, , namevar
    PUT #2, , char0
 
    FOR i = 1 TO ix

       PUT #2, , indx(i)

    NEXT i

    CLOSE #1, #2

