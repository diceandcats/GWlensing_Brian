
=================================================
MACS J0717 HFFv4 reconstruction by the GRALE team
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

RA box limits (arcsec): -109.75 --> 109.75   
Dec box limits (arcsec): -109.75 --> 109.75 
The maps are 219.5 x 219.5 arcsec^2 and 878 x 878 pixels^2, with each pixel = 0.25 arcsec
The center of the reconstructed map is 
      RA = 109.3859632698
      Dec = 37.7512958667

The image set used (37 images from 13 sources) is given below.
Note that only source with z<2.58 were included in this reconstruction becuse some sources at higher z's degrade GRALE fitness values. All image IDs are from Kawamata et al. (2016) unless otherwise noted. Source redshifts were determined spectroscopically for at least one of the images in the system, except for those marked PMave, which means that the redshift we used is an average of the following estimates (whichever are available): photo-z from Kawamata+16, photo-z from Limousin+16, Kawamata+16 model, Limousin+16 three models.
SOURCE.IMAGE    RA               Dec             z     NOTES
 3.1           109.398546        37.741503      1.85
 3.2           109.394459        37.739172      1.85
 3.3           109.407155        37.753830      1.85
 4.1           109.380870        37.750119      1.85
 4.2           109.376438        37.744689      1.85
 4.3           109.391094        37.763300      1.85
 6.1           109.364357        37.757097      2.39
 6.2           109.362705        37.752681      2.39
 6.3           109.373863        37.769703      2.39
 7.1           109.366570        37.766339      1.99   PMave
 7.2           109.365037        37.764119      1.99    "
 7.3           109.359047        37.751781      1.99    "
12.1           109.385165        37.751836      1.71
12.2           109.377617        37.742914      1.71
12.3           109.391219        37.760630      1.71
13.1           109.385674        37.750722      2.55
13.2           109.377564        37.739614      2.55
13.3           109.396212        37.763333      2.55
14.1           109.388791        37.752164      1.85
14.2           109.379664        37.739703      1.85
14.3           109.396192        37.760419      1.85
15.1           109.367663        37.772058      2.40
15.2           109.358624        37.760133      2.40
15.3           109.356540        37.754641      2.40
18.1           109.364249        37.768633      2.21   PMave
18.2           109.361215        37.764333      2.21    "
29.1           109.400879        37.743175      1.63   PMave
29.2           109.392875        37.738603      1.63    "
29.3           109.406088        37.749953      1.63    "
31.1           109.374705        37.756359      1.58   PMave
31.2           109.371020        37.750555      1.58    "
31.3           109.381612        37.764983      1.58    "
24.1           109.392290        37.732950      2.57   PMave Limousin ID
24.2           109.410560        37.748430      2.57    "      "
55.1           109.373760        37.755780      2.33   PMave Limousin ID
55.2           109.370240        37.748740      2.33    "      "
55.3           109.385020        37.768410      2.33    "      "

The spectroscopic redshifts and image identifications quoted above were obtained by these groups:
Kawamata et al. 2016, ApJ, 819, 114  (arXiv:1510.06400) 
Limousin et al. 2016, A&A, 588, A99  (arXiv:1510.08077) 
Diego et al. 2015, MNRAS, 451, 3920  (arXiv:1410.7019) 
Zitrin et al. 2009, ApJL, 707, L102  (arXiv:0907.4232) 

ACKNOWLEDGMENTS

We gratefully acknowledge all the members of the Frontier Fields lens reconstruction teams for data they have contributed to this effort, and without which our mass reconstructions would not have been possible. We are also grateful to Dan Coe for coordinating this project and making it run smoothly. 
 
