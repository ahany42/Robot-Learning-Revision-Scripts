wpts = [1 4 4 3 -2 0 ; 0 1 2 4 3 1]
tpts = [0 1 2 3 4 5]
tvec = 0:0.01:5
[q,qd,qdd,pp] = quinticPolytraj(wpts,tpts,tvec)