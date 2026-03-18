#!/usr/bin/env python

# Simple fusion reactor systems code, based on the approach in Plasma Physics and Fusion Energy, Freidberg
# Section 5.5
# Adapted and expanded by Richard Kembleton for TU/e Fusion Reactor Design Masterclass
# 2021
# Additional plasma equations from Iter Physics Design Guidelines 1989 (Uckan)
# and Iter Physics Basis Nucl. Fusion 39 (1999)
# Radiation equations from
# Matthews et al, Nuc. Fus. 39 (1999)
# Johner, Fus. Sci. Tech. 59 (2011)
# Uckan (1989)

from IPython import embed

from simplesystemcode import InputParameters, simplesystemcode
from utilities import plot_scan, write_csv
## sample script. simplesystemcode function can also be imported into an external script and run for parameter scans, optimization loops, etc. 

# Assumed inputs/targets: have a play! 
input_parameters = InputParameters(
	GrossElecPower=1000.0,
	WallLoad=4.0,
	BMax=13.0,
	SigmaMax=300.0,
	Li6=0.075,
	NeutShield=0.99,
	BlktSupport=0.3,
	ThermalEff=0.4,
	Kappa=1.0,
	PlasmaT=15.0,
	SafetyFac=3.5,
	GamCD=0.5,
	ElectEffCD=0.5,
	PowerRecirc=0.05,
	ZEff=1.0
)

## Do a single run of the systems code
design_point = simplesystemcode(input_parameters)

## sample code for elongation scan
elongations = [1.0,1.1,1.2,1.3,1.4,1.5,1.6]
scan_out = []

for elongation in elongations:
	input_parameters.Kappa = elongation
	scan_out.append(simplesystemcode(input_parameters, print_out=False))


## Sample code for visualising a scan
plot_scan(scan_out, 'Kappa', 'RMajor')

## Sample code for exporting scan data to .csv for Mimer (https://assar.his.se/mimer/html/). Be careful with overwriting data!
write_csv(scan_out, filename = 'demo_output.csv')

## Option to look at scan embedded in IPython, by typing e.g. "scan_out[3]", "scan_out[5]"
# embed()
