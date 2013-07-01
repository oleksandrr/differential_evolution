#!/usr/bin/python

import numpy, time, differential_evolution

from numpy import (exp, sqrt, abs)

def function(vec):
    (a11, a22, b11, b22, b13, a, b, phi, theta) = vec
    
    O2 = 0.235; O1 = 72;
    E2 = 0.031; E1 = 0.87;
    
    O4 = 0.0222196; O3 = 0.958157;
    E4 = 0.006198; E3 = 0.018862;
    
    O6 = 0.0941698; O5 = 1.60087;
    E6 = 3.6079e-7; E5 = 0.00026;
    
    var3 = E1**2; var4 = 1/var3; var5 = -2*O1; var6 = b11 + b22; var7 = 1j*phi;
    var8 = exp(var7); var9 = var6*var8; var10 = -a22; var11 = a11 + var10; 
    var12 = var11**2; var13 = -b22; var14 = b11 + var13; var15 = 2*var11*var14*var8;
    var16 = b13**2; var17 = 4*var16; var18 = var14**2; var19 = var17 + var18; 
    var20 = 2j*phi; var21 = exp(var20); var22 = var19*var21;
    var23 = var12 + var15 + var22; var24 = sqrt(var23); var25 = -var24; 
    var26 = a11 + a22 + var9 + var25; var27 = abs(var26); var28 = var5 + var27;
    var29 = var28**2; var31 = E2**2; var32 = 1/var31; var33 = -2*O2; 
    var34 = a11 + a22 + var9 + var24; var35 = abs(var34); var36 = var33 + var35;
    var37 = var36**2; var39 = E3**2; var40 = 1/var39; var41 = -2*O3; 
    var42 = a11 + a22; var43 = a*var42; var44 = 1j*theta; var45 = exp(var44);
    var46 = b*var6*var45; var47 = a**2; var48 = var47*var12;
    var49 = 2*a*var11*b*var14*var45; var50 = b**2; var51 = 2j*theta;
    var52 = exp(var51); var53 = var50*var19*var52; var54 = var48 + var49 + var53; 
    var55 = sqrt(var54); var56 = -var55; var57 = var43 + var46 + var56;
    var58 = abs(var57); var59 = var41 + var58; var60 = var59**2; var62 = E4**2; 
    var63 = 1/var62; var64 = -2*O4; var65 = var43 + var46 + var55;
    var66 = abs(var65); var67 = var64 + var66; var68 = var67**2; var70 = E5**2; 
    var71 = 1/var70; var72 = -2*O5; var73 = -3*b*var6*var45;
    var74 = -6*a*var11*b*var14*var45; var75 = 9*var50*var19*var52;
    var76 = var48 + var74 + var75; var77 = sqrt(var76); var78 = -var77;
    var79 = var43 + var73 + var78; var80 = abs(var79); var81 = var72 + var80;
    var82 = var81**2; var84 = E6**2; var85 = 1/var84; var86 = -2*O6;
    var87 = var43 + var73 + var77; var88 = abs(var87); var89 = var86 + var88;
    var90 = var89**2; 
    
    return (var4*var29 + var32*var37 + var40*var60 +
            var63*var68 + var71*var82 + var85*var90)/4

t_start = time.clock()

(iterations, (costs, population)) = differential_evolution.minimize(
                                    function, numpy.random.uniform(-1.0, 1.0, (50, 9)),
                                    0.850, 0.975, 1.0e-12, 250, 12500,
                                    mutation_method="MDE5", selection_method="Storn-Price",
                                    output_function=None
                                   )

t_total = time.clock() - t_start

print "Performed", iterations, "iterations, taking", t_total, "seconds."

numpy.set_printoptions(precision=16,linewidth=109)
best_index = numpy.argmin(costs)
print "Optimized function value:", costs[best_index]
print "Optimized parameter vector:\n", population[best_index]
