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

import numpy as np
from IPython import embed
# Assumed limits / targets - have a play!

def simplesystemcode(print_out=True, **kwargs):

	inputs = {}
	# Assumed inputs/targets: have a play! 
	# Values below are defaults - they can be altered by the **kwargs input directory
	inputs["GrossElecPower"] = 1000.0 # Target gross electrical power, MW
	inputs["WallLoad"] = 4.0          # Neutron wall load limit, MW m-2
	inputs["BMax"] = 13.0             # Maximum field on the TF superconductor, T
	inputs["SigmaMax"] = 300.0        # Maximum stress on the TF structure, MP
	inputs["Li6"] = 0.075             # Fractional concentration of Li6 in breeder material
	inputs["NeutShield"] = 0.99       # Neutron shielding efficiency target for blanket, VV
	inputs["BlktSupport"] = 0.3       # Blanket/shield structural support thickness, m
	inputs["ThermalEff"] = 0.4        # Thermal efficiency of electricity generation, Pe/Pth
	inputs["Kappa"] = 1.0             # Plasma Elongation
	inputs["PlasmaT"] = 15.0          # Plasma temperature, keV (probably shouldn't change this without a better SigV function)
	inputs["SafetyFac"] = 3.5         # Plasma safety factor (q_edge)
	inputs["GamCD"] = 0.5             # Current drive efficiency, gamma_CD
	inputs["ElectEffCD"] = 0.5        # Electrical efficiency of CD system
	inputs["PowerRecirc"] = 0.05      # Fraction of extracted heat representing coolant pumping power
	inputs["ZEff"] = 1.00             # Plasma Zeff - note plasma dilution is not calculated, nor He checked! For radiation calculation


	input_names = ['GrossElecPower','WallLoad','BMax','SigmaMax','Li6','NeutShield','BlktSupport','ThermalEff','Kappa',\
    'PlasmaT','SafetyFac','GamCD','ElectEffCD','PowerRecirc','ZEff']

    # check sanity of inputs
	for key in kwargs.keys():
		if key not in input_names:
			raise NameError(key+" is not an allowed input variable. Check for possible typo")
		inputs[key] = kwargs[key]

	# Physical inputs - don't alter
	WallRef = 0.6           # Wall reflectivity for synchrotron radiation
	SigV = 3.00e-22         # Velocity-averaged reactivity for fusion reaction at 15keV, m3 s-1
	NeutFastXC = 4.0        # Fast neutron slowing cross-section, barns !!! Higher than Freidberg value
	NeutBreedXC = 950.0     # Slow neutron breeding cross-section, barns
	LiDens = 4.50e28        # Number density of Li atoms, m-3
	EFastNeut = 14.1        # Fast neutron energy, MeV
	EThNeut = 0.025         # Thermal neutron energy, eV
	Mu0 = 1.25664e-6        # Magnetic permeability, H m-1

	# Derived inputs, reverse-engineered from values in Freidberg
	EnRatioOne = 0.631654682 # En/(Ea+En+ELi)
	EnRatioTwo = 0.784      # (En+Ea)/(Ea+En+ELi)

	# Step 1: blanket and shield thickness
	NeutronMFP = 1.0/(LiDens * NeutFastXC * 1.0e-28) # Freidberg lambda_sd, pag 94
	Li6Dens = inputs["Li6"] * LiDens # Freidberg n, pag 96
	BreedLength = 1.0/(Li6Dens * NeutBreedXC * 1.0e-28) # Freidberg lambda_br, pag 96
	DelX = 2.0 * NeutronMFP * np.log(1.0 - 0.5*np.sqrt(1.0e6*EFastNeut/EThNeut)*(BreedLength/NeutronMFP)*np.log(1.0-inputs["NeutShield"])) # Freidberg eq. 5.10
	BlnkThk = DelX + inputs["BlktSupport"]    # Total blanket thickness, m

	# Step 2: Plasma radius and coil thickness
	CoilZeta = (inputs["BMax"]**2)/(4.0 * Mu0 * inputs["SigmaMax"] * 1.0e6) # Freidberg ksi, pag 100
	VIoPe = (1.0/inputs["ThermalEff"]) * EnRatioOne * np.sqrt(inputs["Kappa"]) * (BlnkThk/inputs["WallLoad"]) * ((1.0+CoilZeta)/((1.0-np.sqrt(CoilZeta))**2)) # Freidberg eq. 5.30, constant from 5.21	
	RMinor = ((1.0 + CoilZeta)/(2.0 * np.sqrt(CoilZeta)))*BlnkThk   # Plasma minor radius Freidberg eq. 5.29, m
	R0a = EnRatioOne * (inputs["GrossElecPower"]/inputs["WallLoad"])
	R0b = 1.0/(inputs["ThermalEff"]*4.0*np.pi*np.pi*np.sqrt(inputs["Kappa"]))
	RMajor = R0a * R0b / RMinor     # Plasma major radius Freidberg eq. 5.20, m
	PlasVol = 2.0 * np.pi**2 * RMinor**2 * inputs["Kappa"] * RMajor   # Plasma Volume, m3

	# Others
	PressPrefix = 0.0000000000084 * np.sqrt(400)
	Pressure = PressPrefix * np.sqrt((inputs["PlasmaT"]**2)/(SigV * PlasVol))  # Plasma pressure, from Freidberg eq. 5.37, atm
	Aspect = RMajor/RMinor  # Plasma aspect ratio
	InvAspect = 1.0/Aspect # Inverse aspect ratio
	MagField = inputs["BMax"] * (RMajor - RMinor - BlnkThk)/RMajor   # Magnetic field in the plasma, Freidberg eq. 5.42, T
	
	#!!!! Find sources:
	 GeoFac = (1.17 - 0.65*InvAspect)/((1.0-InvAspect**2)**2)
	 PlasCur = GeoFac * (5.0 * RMinor**2 * MagField)/(RMajor * inputs["SafetyFac"]) * (1.0 + inputs["Kappa"]**2)/2.0 # Assumes no triangularity for simplicity, includes Freidberg 14.165 ???
	 BPol = Mu0 * PlasCur * 1.0e6 / (2.0 * np.pi * RMinor * np.sqrt(inputs["Kappa"]))
	BetaPol = 2.0 * 100000.0 * Pressure * Mu0/(BPol**2)  # Plasma poloidal beta (normalised plasma pressure), Freidberg eq. 5.1
	 BootFrac = min(0.5 * np.sqrt(InvAspect) * BetaPol, 1.0) # Bootstrap fraction, very simple formula
	DrivenCurrent = PlasCur * (1.0 - BootFrac)
	 PlasmaDens = Pressure/(2.0*0.1602*inputs["PlasmaT"])      # Plasma electron density, 10^20 m-3
	
	#!!!! Get sankey diagram source
	PowerCD = PlasmaDens * DrivenCurrent * RMajor / inputs["GamCD"] # MW
	PowerCDElec = PowerCD/inputs["ElectEffCD"] # Electrical power for current drive, MW
	ThermalPow = inputs["GrossElecPower"] / inputs["ThermalEff"] # Gross thermal power
	CoolantPower = ThermalPow * inputs["PowerRecirc"] # Power needed for coolant pumps
	NetElecPower = inputs["GrossElecPower"] - PowerCDElec - CoolantPower # Net electrical power once the other two are taken into account

	FusPDens = EnRatioTwo*inputs["GrossElecPower"]/(inputs["ThermalEff"]*PlasVol)   # FusionPowerDensity, MW m-3
	FusPower = FusPDens * PlasVol   # Fusion Power, MW
	PlasHeat = 0.2 * FusPower + PowerCD # Total plasma heating power, MW
	PTauE = 0.0562 * PlasCur**0.93 * MagField**0.15 * (10.0*PlasmaDens)**0.41 * PlasHeat**(-0.69) * RMajor**1.97 * inputs["Kappa"]**0.78 * InvAspect**0.58 * 2.5**0.19 # Prediction from IPB98(y,2), s
	ThermalE = Pressure * PlasVol / 10.0 # Thermal energy, MJ
	TauE = ThermalE/PlasHeat
	HFact = TauE/PTauE

	# Final parameters
	MagThk = (np.sqrt(CoilZeta)*(1.0+np.sqrt(CoilZeta)))/(1.0-np.sqrt(CoilZeta))*BlnkThk    # Magnet thickness, Freidberg eq. 5.27 & 5.29, m
	Beta = 2.0 * 100000.0 * Pressure * Mu0/(MagField**2)  # Plasma beta (normalised plasma pressure), Freidberg eq. 5.1
	PlasSurf = 4.0 * np.pi**2 * RMinor * RMajor * np.sqrt(inputs["Kappa"])  # Plasma surface area, m2
	WallLoadCalc = 0.8 * FusPower / PlasSurf  # Cross-check on neutron wall loading, MW m-2
	StableKappa = 1.46 + 0.5/(Aspect-1.0)  # Estimated maximum vertically-controllable kappa
	nG = PlasCur / (np.pi * RMinor**2) # Greenwald density limit, Freidberg eq. 14.146

	# Radiation losses
	# Using forms from Matthews et al Nuc. Fus. 39 (1999)
	# Johner, Fus. Sci. Tech 59 (2011)
	# ITER Physics Basics (1989)

	LineRad = ((inputs["ZEff"]-1.0) * PlasSurf**0.94 * PlasmaDens**1.8)/4.5  # Line radiation
	BremRad = 5.355e-3 * inputs["ZEff"] * PlasmaDens**2.0 * inputs["PlasmaT"]**0.5 * PlasVol  # Bremsstrahlung radiation
	Lambda0 = 77.7 * ((PlasmaDens * RMinor)/MagField)**0.5 # Lambda factor for synchrotron radiation
	GSyn = 0.16 * (inputs["PlasmaT"]/10.0)**1.5 * (1.0 + 5.7/(Aspect * (inputs["PlasmaT"]/10.0)**0.5))**0.5 # G factor for synchrotron radiation
	SynchRad = 6.2e-2 * (GSyn/Lambda0) * (1.0-WallRef)**0.5 * PlasmaDens * (inputs["PlasmaT"]/10.0) * MagField**2.0 * PlasVol # Synchrotron radiation
	DivLoad = (PlasHeat-LineRad-BremRad-SynchRad) * MagField / (Aspect * RMajor * inputs["SafetyFac"]) # Divertor heat load factor ~ 9.2 for EU-DEMO

	out_dict = {}
	# Set output dictionary
	out_dict["RMajor"] = RMajor
	out_dict["RMinor"] = RMinor
	out_dict["Aspect"] = Aspect
	out_dict["Kappa"] = inputs["Kappa"]
	out_dict["StableKappa"] = StableKappa  # Ideally the plasma elongation is lower than this
	out_dict["BootFrac"] = BootFrac # The higher, the better
	out_dict["PlasVol"] = PlasVol # 
	out_dict["PlasSurf"] = PlasSurf # 
	out_dict["FusPower"] = FusPower
	out_dict["PowerCD"] = PowerCD
	out_dict["PowerCDElec"] = PowerCDElec
	out_dict["CoolantPower"] = CoolantPower
	out_dict["NetElecPower"] = NetElecPower
	out_dict["HFact"] = HFact  # Ideally not too much above 1.2 or so
	out_dict["n_nG"] = PlasmaDens/nG  # Ideally not too much above 1.2 or so
	out_dict["DivLoad"] = DivLoad # Probably try to keep this below 30 for the purposes of this exercise
	out_dict["MagField"] = MagField
	out_dict["betaN"] = 100.0 * Beta * RMinor * MagField/PlasCur # Below 4.5 or so
	out_dict["LineRad"] = LineRad
	out_dict["BremRad"] = BremRad
	out_dict["SynchRad"] = SynchRad
	out_dict["ZEff"] = inputs["ZEff"] # If this is too high the plasma is radiatively unstable, and highly diluted so fusion power will be depressed (although improvement of confinement with Zeff somewhat compensates for this)
	out_dict["MagThk"] = MagThk
	out_dict["BlnkThk"] = BlnkThk
	out_dict["Bore"] = RMajor - RMinor - BlnkThk - MagThk # Affects the size of cental solenoid and hence flux swing available
	out_dict["WallLoadCalc"] = WallLoadCalc

	# Print output
	if print_out:
		print("Simple systems code fusion power plant:\n")
		print('Major radius: {:2.2f} m'.format(RMajor))
		print('Minor radius: {:2.2f} m'.format(RMinor))
		print('Aspect ratio: {:2.2f}'.format(Aspect))
		print('Plasma elongation: {:2.2f}'.format(inputs["Kappa"]))
		print('Stable elongation: {:2.2f}'.format(StableKappa))   # Ideally the plasma elongation is lower than this
		print('Bootstrap current fraction: {:2.2f}'.format(BootFrac))   # The higher, the better
		print('Plasma volume: {:2.2f} m3'.format(PlasVol))
		print('Plasma surface area: {:2.2f} m2'.format(PlasSurf))
		print("")
		print('Fusion power: {:2.2f} MW'.format(FusPower))
		print('CD power for steady-state: {:2.2f} MW'.format(PowerCD))
		print('CD electrical power: {:2.2f} MW'.format(PowerCDElec))
		print('Coolant pumping power: {:2.2f} MW'.format(CoolantPower))
		print('Net elec power: {:2.2f} MW'.format(NetElecPower))
		print("")
		print('H factor: {:2.2f} (IPB98(y,2))'.format(HFact))       # Ideally not too much above 1.2 or so
		print('n / nG: {:2.2f}'.format(PlasmaDens/nG))              # Ideally not too much above 1.2 or so
		print('Divertor power loading: {:2.2f} MW T m-1'.format(DivLoad))  # Probably try to keep this below 30 for the purposes of this exercise
		print('Field in plasma: {:2.2f} T'.format(MagField))
		print('Normalised beta: {:2.2f} % m T MA-1'.format(100.0 * Beta * RMinor * MagField/PlasCur))   # Below 4.5 or so
		print('Line radiation: {:2.2f} MW'.format(LineRad))
		print('Bremsstrahlung radiation: {:2.2f} MW'.format(BremRad))
		print('Synchrotron radiation: {:2.2f} MW'.format(SynchRad))
		print('Plasma ZEff: {:2.2f}'.format(inputs["ZEff"]))  # If this is too high the plasma is radiatively unstable, and highly diluted so fusion power will be depressed (although improvement of confinement with Zeff somewhat compensates for this)
		print("")
		print('Magnet thickness: {:2.2f} m'.format(MagThk))
		print('Blanket/shield thickness: {:2.2f} m'.format(BlnkThk))
		print('Bore: {:2.2f} m'.format(RMajor - RMinor - BlnkThk - MagThk))   # Affects the size of cental solenoid and hence flux swing available
		print('Wall load: {:2.2f} MW m-2'.format(WallLoadCalc))

	return out_dict


# sample script. simplesystemcode function can also be imported into an external script and run for parameter scans, optimization loops, etc. 

in_dict = {}

# Assumed inputs/targets: have a play! 
in_dict["GrossElecPower"] = 1000   # 1000.0 # Target gross electrical power, MW
in_dict["WallLoad"] = 4.0          # 4.0   # Neutron wall load limit, MW m-2    # if larger, then R can be smaller - Eq 5.20
in_dict["BMax"] =  13.0  #13.0             # Maximum field on the TF superconductor, T
in_dict["SigmaMax"] = 300   #300.0        # Maximum stress on the TF structure, MPa
in_dict["Li6"] = 0.075 # 0.075             # Fractional concentration of Li6 in breeder material
in_dict["NeutShield"] = 0.99 # 0.99       # Neutron shielding efficiency target for blanket, VV
in_dict["BlktSupport"] = 0.3 # 0.3       # Blanket/shield structural support thickness, m
in_dict["ThermalEff"] = 0.4 #0.4        # Thermal efficiency of electricity generation, Pe/Pth
in_dict["Kappa"] = 1.0   # 1.0             # Plasma Elongation
in_dict["PlasmaT"] = 15.0 #15.0          # Plasma temperature, keV (probably shouldn't change this without a better SigV function)
in_dict["SafetyFac"] = 3.5 #  3.5         # Plasma safety factor (q_edge)
in_dict["GamCD"] = 0.5 #  0.5             # Current drive efficiency, gamma_CD
in_dict["ElectEffCD"] = 0.5 # 0.5        # Electrical efficiency of CD system
in_dict["PowerRecirc"] = 0.05 # 0.05      # Fraction of extracted heat representing coolant pumping power
in_dict["ZEff"] = 1.0 # 1.0              # Plasma Zeff - note plasma dilution is not calculated, nor He checked! For radiation calculation


out_dict = simplesystemcode(**in_dict)

## sample code for elongation scan

# elongations = [1.0,1.1,1.2,1.3,1.4,1.5]
# scan_out = []

# for elongation in elongations:
# 	in_dict["Kappa"] = elongation
# 	scan_out.append(simplesystemcode(**in_dict, print_out=False))

# embed()   # now, you will be embedded in IPython and can look at the scan by typing e.g. "scan_out[3]", "scan_out[5]"


#### TODO:
# - Sankey diagram?
# - Kappa definition, why adjustments to various equations?