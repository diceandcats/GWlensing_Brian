

Abell 370 (z=0.375)
-------------------

v4.0 (all images) refers to products delivered by September 15, 2017

Authors: Liliya Williams, Kevin Sebesta, Jori Liesenborgs

------------------------------------------------------------------------------------------
BRIEF DESCRIPTION OF THE METHOD: GRALE

The method used to do the strong lensing reconstruction is GRALE, a free-form, adaptive grid 
method that uses a genetic algorithm to iteratively refine the mass map solution. GRALE does 
not use any information from the cluster or cluster galaxies to do the mass reconstruction; 
the only observational input are the lensed image positions and their redshifts.

An initial course grid is populated with a basis set, consisting of projected Plummer density 
profiles. A uniform mass sheet covering the whole modeling region can also be added to 
supplement the basis set, but is not used here. As the code runs the more dense regions are 
resolved with a finer grid, with each cell given a Plummer with a proportionate width. The 
code is started with an initial set of trial solutions. These solutions, as well as all the 
later evolved ones are evaluated for genetic fitness, and the fit ones are cloned, combined 
and mutated. The procedure is run until a satisfactory degree of mass resolution is achieved.  
The final map consists of a superposition of many Plummers, typically 1000-3000, each with its 
own size and weight, determined by the genetic algorithm. In this release we present 40 such
maps, with each run using its own random seed. The best map is the average mass distribution 
of the 40 maps, and consists of 84913 Plummers. 

The dispersion between the 40 maps is the uncertainty. Because GRALE explores a wide range of 
lensing degeneracies, the estimated uncertainties are usually larger than in some other methods. 
(Lensing degeneracies means that many different mass maps can reproduce the same lensed images 
equally well.)
	
Sebesta, K., Williams, L. L. R., Mohammed, I., Saha, P. & Liesenborgs, J. 2016, MNRAS, 461, 2126
"Testing light-traces-mass in Hubble Frontier Fields Cluster MACS-J0416.1-2403"

Mohammed, I., Liesenborgs, P. Saha, & L.L.R. Williams. 2013, MNRAS, 2016, MNRAS, 459, 1698
"Mass-Galaxy offsets in Abell 3827, 2218 and 1689: intrinsic properties or line-of-sight substructures?"

Liesenborgs, J., de Rijcke, S., Dejonghe, H. & Bekaert, P. 2007, MNRAS, 380, 1729
"Non-parametric inversion of gravitational lensing systems with few images using a 
multi-objective genetic algorithm"

Liesenborgs, J., de Rijcke, S. & Dejonghe, H. 2006, MNRAS, 367, 1209
"A genetic algorithm for the non-parametric inversion of strong lensing systems"

------------------------------------------------------------------------------------------
DELIVERABLES

The directory contains 40 individual reconstructions, and the best (average) map.
For each one of these 41 maps, we upload 11 separate FITS files for kappa, gamma_tot, 
gamma_x, gamma_y, potential, deflection angles in the x andy directions 
(for sources at "infinity"), and magnification maps for sources at z=1, 2, 4, and 9. 

(0,0) of the map corresponds to the center of the field at
      RA = 39.9724458375 degrees
      Dec= -1.5770694440 degrees

Each map covers -59.8" --> 89.8" in RA, and -79.8" --> 69.8" in Dec (w.r.t. to the center),
and has 749x749 pixels; each pixel is 0.2".

We use the standard cosmology: Omega_m=0.3, Lambda=0.7, h=0.7.

INPUT SOURCES USED:

All image set: 124 images from 41 sources  

 Name    delta RA   delta Dec    z
  1.1   -10.50162     0.33991  0.8041
  1.2    22.63824     3.44992  0.8041
  1.3    -4.74380     1.45016  0.8041
  2.1    13.85031   -26.00993  0.7251
  2.2     3.42869   -28.96984  0.7251
  2.3    -4.51707   -27.06976  0.7251
  2.4    -2.07360   -27.83980  0.7251
  2.5    -1.27831   -28.23004  0.7251
  3.1   -15.65854    36.51990  1.9553
  3.2    -5.29802    40.28012  1.9553
  3.3    32.17836    34.36093  1.9553
  4.1    34.72246     2.21004  1.2728
  4.2     2.59022     2.63024  1.2728
  4.3   -29.02019    -3.36994  1.2728
  5.1    12.47201   -43.37993  1.2774
  5.2     3.71298   -44.00992  1.2774
  5.3    -1.33948   -43.62004  1.2774
  5.4    -5.11803   -43.13008  1.2774
  6.1    -2.07361    -0.74008  1.0633
  6.2   -20.41225    -4.43011  1.0633
  6.3    34.70805    -0.33012  1.0633
  7.1    -0.81408   -12.35008  2.7512
  7.2    -0.46862   -13.57984  2.7512
  7.3    -4.29395   -31.07980  2.7512
  7.4    59.56742    -2.09032  2.7512
  7.5   -30.44162   -10.82014  2.7512
  8.1   -19.90132    25.83005  2.042 
  8.2   -29.15340    12.09998  2.042 
  9.1   -27.35402    -3.18993  1.5182
  9.2    -1.86489     2.63996  1.5182
  9.3    43.24043     1.68007  1.5182
 10.1    -5.16125    18.86012  2.7512
 10.2    -7.18729    22.05008  2.7512
 10.3    -3.46988    -1.72000  2.7512
 11.1   -22.30163    27.50009  4.667 
 11.2   -33.21625    10.27009  4.667 
 11.3    66.43020    18.11643  4.667 
 12.1    -1.45825    37.35008  3.4809
 12.2   -38.79413     6.32014  3.4809
 12.3    50.79771    22.11003  3.4809
 13.1    34.31589    18.75996  4.2467
 13.2    18.63302    29.43006  4.2467
 13.3   -47.67552    -1.82095  4.2467
 14.1     8.21129    -3.53981  3.1277
 14.2     7.88381   -11.16965  3.1277
 14.3    15.04864   -30.98982  3.1277
 14.4    40.70696    -4.16991  3.1277
 14.5   -44.37191   -12.40997  3.1277
 15.1     4.77459   -12.97432  3.7085
 15.2     6.95894   -36.18424  3.7085
 15.3     3.69141    -2.84644  3.7085
 15.4    50.43766    -5.22513  3.7085
 15.5   -40.55016   -12.74763  3.7085
 16.1   -21.53852   -39.88147  3.7743
 16.2   -14.26571   -43.32774  3.7743
 16.3    51.86618   -25.60006  3.7743
 17.1    -0.87526   -41.51980  4.2567
 17.2    55.42530   -13.82629  4.2567
 17.3   -35.14497   -23.94288  4.2567
 18.1    20.97556   -36.22063  4.4296
 18.2    41.29346   -18.26212  4.4296
 18.3   -45.48385   -18.31002  4.4296
 19.1     7.17845   -39.11536  5.6493
 19.2    54.48611    -7.53996  5.6493
 19.3   -42.05077   -15.51352  5.6493
 20.1   -17.02226   -38.89002  5.7505
 20.2   -23.00677   -35.57984  5.7505
 21.1   -12.32968   -27.40997  1.2567
 21.2   -10.02      -28.58     1.2567 
 21.3    41.52019   -15.85012  1.2567
 22.1    15.85833   -32.75994  3.1277
 22.2    42.00963    -9.67000  3.1277
 22.3   -43.52622   -14.43893  3.1277
 23.1    36.89621    36.84995  5.9386
 23.2   -45.65679    15.32154  5.9386
 23.3    25.77997    38.61076  5.9386
 24.1   -24.78828    23.06020  4.9153
 24.2   -28.68919    16.79006  4.9153
 25.1    62.34193    -6.72030  3.8084
 25.2   -28.97333   -22.20982  3.8084
 25.3   -10.85783   -35.27957  3.8084
 26.1    35.76255    20.19527  3.9359
 26.2    16.05999    32.06238  3.9359
 26.3   -46.19289     0.14934  3.9359
 27.1    38.47234    21.17986  3.0161
 27.2   -42.14445     3.93548  3.0161
 27.3     8.78352    35.42011  3.0161
 28.1   -23.42426   -19.01012  2.9112
 28.2   -10.59154   -27.20981  2.9112
 28.3    64.10890    -1.63028  2.9112
 29.1    -6.84184    43.83008  4.4897
 29.2    48.76095    33.99292  4.4897
 29.3   -32.72332    28.73017  4.4897
 30.1    48.42981    23.58280  5.6459
 30.2     8.64677    38.32819  5.6459
 31.1     8.64676    27.61171  5.4476
 31.2    38.38230     8.15542  5.4476
 31.3   -49.81668    -6.04305  5.4476
 32.1   -13.36978    27.55951  4.4953
 32.2    65.12379     6.52624  4.4953
 32.3   -33.53647    -4.96620  4.4953
 33.1   -26.19156   -32.41257  4.882 
 33.2   -13.61796   -39.58589  4.882 
 34.1     0.38426    24.66044  5.2437
 34.2     6.49471   -39.74212  5.2437
 34.3   -41.15471   -16.92291  5.2437
 34.4    54.14063    -9.21252  5.2437
 35.1    41.52769    40.09533  6.1735
 35.2    20.96139    45.20777  6.1735
 36.1   -27.19564   -13.35525  6.2855
 36.2   -14.41328   -26.58378  6.2855
 37.1     1.39908    29.54060  5.6489
 37.2     1.53583    27.28700  5.6489
 38.1    25.74029    11.54991  3.1563
 38.2    18.24434    17.58210  3.1563
114.1   -16.40697    -3.67986  1.2777
114.2    -7.44277    -1.24984  1.2777
114.3    44.24444     0.08995  1.2777
115.1   -23.11136    40.92016  1.0323
115.2   -25.34612    39.20008  1.0323
115.3   -23.84549    39.76996  1.0323
122.1   -28.37247     6.30003  2.0   
122.2    50.57093    13.11975  2.0   
122.3   -12.37295    26.39023  2.0   


The image information was taken from
Caminha et al. 2016
Karman et al. 2017
Diego et al. 2016
Lagattuta et al. 2016, 2017

Acknowledgements:

The GRALE team gratefully acknowledges the contribution of the other HFF team members to
this effort, and especially Keren Sharon, Dan Coe, Jesus Vega, Jose (Chema) Diego, and 
David Lagatutta.

The GRALE team would like to thank the Minnesota Supercomputing Institute for the use of
their resources and technical support.

