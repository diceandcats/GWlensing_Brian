
=================================================
MACS J1149 HFFv4 reconstruction by the GRALE team
=================================================

Authors: Liliya Williams, Kevin Sebesta, Jori Liesenborgs

BRIEF DESCRIPTION OF THE METHOD: GRALE

The method used to do the lensing reconstruction is GRALE, a free-form, adaptive grid method  that uses a genetic algorithm to iteratively refine the mass map solution. 

An initial course grid is populated with a basis set, such as a projected Plummer density profiles. A uniform mass sheet covering the whole modeling region can also be added to supplement the basis set.  As the code runs the more dense regions are resolved with a finer grid, with each cell given a Plummer with a proportionate width. The code is started with an initial set of trial solutions. These solutions, as well as all the later evolved ones are evaluated for genetic fitness, and the fit ones are cloned, combined and mutated. The procedure is run until a satisfactory degree of mass resolution is achieved.  The final map consists of a superposition of a mass sheet and many Plummers, typically several 100 to 1-2 thousand, each with its own size and weight, determined by the genetic algorithm. In this release we present 40 independent realizations of the reconstruction for each cluster.

Note that GRALE does not use any cluster galaxies to do the mass reconstruction; the only observational input are the lensed image positions, their redshifts (listed above), the redshift of the lensing cluster, and the parameters of the standard Lambda CDM.

For more detailed description of the method see

Priewe, J., Williams, L.L.R., Liesenborgs, J., Coe, D. & Rodney, S. A.	2017, MNRAS, 465, 1030
"Lens Models Under the Microscope: Comparison of Hubble Frontier Field Cluster Magnification Maps"

M. Meneghetti et al. 
"The Frontier Fields Lens Modeling Comparison Project"
	
Sebesta, K., Williams, L.L.R., Mohammed, I., Saha, P. & Liesenborgs, J.
"Testing light-traces-mass in Hubble Frontier Fields Cluster MACS-J0416.1-2403"

Liesenborgs, J., de Rijcke, S., Dejonghe, H., Bekaert, P. 2009, MNRAS, 397, 341
"Non-parametric strong lens inversion of SDSS J1004+4112"

Liesenborgs, J., de Rijcke, S., Dejonghe, H. & Bekaert, P. 2007, MNRAS, 380, 1729
"Non-parametric inversion of gravitational lensing systems with few images using a 
multi-objective genetic algorithm"

Liesenborgs, J., de Rijcke, S. & Dejonghe, H. 2006, MNRAS, 367, 1209
"A genetic algorithm for the non-parametric inversion of strong lensing systems"


COSMOLOGY

Omega_matter=0.3
Omega_Lambda=0.7
H_0=70 km/s/Mpc

DELIVERABLES

FITS maps with Dls/Ds=1: kappa, gamma, gamma1, gamma2, deflect_x (per pixel), deflect_y (per pixel) 
FITS maps of magnifications for sources at z = 1, 2, 4, 9

RA box limits (arcsec): -69.8 --> 69.8   
Dec box limits (arcsec): -69.8 --> 69.8 
The maps are 139.6 x 139.6 arcsec^2 and 698 x 698 pixels^2, with each pixel = 0.2 arcsec
The center of the reconstructed map is 
      RA = 177.40291043716
      Dec = 22.3985747135

The image set used (59 images from 20 sources) is given below
(source redshifts were determined spectroscopically for at least one of the images in the system, unless otherwise noted).
SOURCE.IMAGE    RA               Dec             z       NOTES
1.1             177.397          22.396         1.488 
1.2             177.39942        22.397439      1.488
1.3             177.40342        22.402439      1.488
1b.1            177.39672        22.395352      1.488
1b.2            177.39977        22.397498      1.488
1b.3            177.40324        22.402         1.488
1c.1            177.39815        22.396341      1.488
1c.2            177.39926        22.396833      1.488
1c.3            177.40383        22.40255       1.488
2.1             177.40242        22.38975       1.893
2.2             177.40604        22.392478      1.893
2.3             177.40658        22.392886      1.893
3.1             177.39075        22.399847      3.129
3.2             177.39271        22.403081      3.129
3.3             177.40129        22.407189      3.129
4.1             177.393          22.396825      2.949
4.2             177.39438        22.400736      2.949
4.3             177.40417        22.406128      2.949
5.1             177.39975        22.393061      2.8
5.2             177.40108        22.393825      2.8
5.3             177.40792        22.403553      2.8
6.1             177.39971        22.392544      3.04   model z from Johnson+2014 ApJ...797...48 (1405.0222)
6.2             177.40183        22.393858      3.04       "
6.3             177.40804        22.402506      3.04       "
7.1             177.39896        22.391339      3.06   model z from Johnson+2014 ApJ...797...48 (1405.0222)
7.2             177.40342        22.394269      3.06       "
7.3             177.40758        22.401242      3.06       "
8.1             177.3985         22.39435       2.784
8.2             177.39979        22.395044      2.784
8.3             177.40704        22.405553      2.784
8.4             177.407092       22.404722      2.784
12.1            177.3985         22.389356      3.8    
12.2            177.40375        22.392345      3.8      
12.3            177.40822        22.398801      3.8       
13.1            177.40371        22.397786      1.241  
13.2            177.40283        22.396656      1.241
13.3            177.40004        22.393858      1.241
14.1            177.39167        22.403489      3.703
14.2            177.39083        22.402647      3.703
24.1            177.39285        22.412872      2.48   source #21 in Jauzac+2015 MNRAS.457.2029 (1509.08914)
24.2            177.39353        22.413071      2.48      "
24.3            177.39504        22.412697      2.48      "
28.1            177.39531        22.391809      2.62   model z from Sharon/Johnson (Google spreadsheet)
28.2            177.40215        22.39675       2.62      "
28.3            177.40562        22.402434      2.62      "
200.1           177.408746       22.394467      2.319
200.2           177.405117       22.391261      2.319
200.3           177.402558       22.389233      2.319
201.1           177.400483       22.395444      2.784  same galaxy complex as s.8
201.2           177.406829       22.404517      2.784     "
203.1           177.409946       22.387244      4.8    photo-z
203.2           177.406571       22.384511      4.8      "
203.3           177.411229       22.388461      4.8      "
204.1           177.409608       22.386661      6.5    photo-z
204.2           177.406679       22.384322      6.5      "
204.3           177.412079       22.389056      6.5      "
205.1           177.405196       22.386042      3.4    source #34 in Jauzac+2015 MNRAS.457.2029 (1509.08914)
205.2           177.408208       22.388119      3.4      "
205.3           177.410375       22.390625      3.4      "

The spectroscopic redshifts and image identifications quoted above were obtained by these groups:
Jauzac et al. 2016, MNRAS, 457, 2029  (arXiv:1509.08914)
Sharon et al. 2015, ApJ, 800, L26  (arXiv:1411.6933) 
Smith et al. 2009, ApJ, 707, L163  (arXiv:0911.2003) 
Treu et al. 2016, ApJ, 817, 60  (arXiv:1510.05750)

ACKNOWLEDGMENTS

We gratefully acknowledge all the members of the Frontier Fields lens reconstruction teams for data they have contributed to this effort, and without which our mass reconstructions would not have been possible. We are also grateful to Dan Coe for coordinating this project and making it run smoothly. 
 
