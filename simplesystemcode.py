from dataclasses import dataclass
import numpy as np
import math


@dataclass
class InputParameters:

	# Values below are defaults - they can be altered when calling a simplesystemcode
	GrossElecPower: float = 1000.0 # Target gross electrical power, MW
	WallLoad: float = 4.0          # Neutron wall load limit, MW m-2
	BMax: float = 13.0             # Maximum field on the TF superconductor, T
	SigmaMax: float = 300.0        # Maximum stress on the TF structure, MPa
	Li6: float = 0.075             # Fractional concentration of Li6 in breeder material
	NeutShield: float = 0.99       # Neutron shielding efficiency target for blanket, VV
	BlktSupport: float = 0.3       # Blanket/shield structural support thickness, m
	ThermalEff: float = 0.4        # Thermal efficiency of electricity generation, Pe/Pth
	Kappa: float = 1.0             # Plasma Elongation
	PlasmaT: float = 15.0          # Plasma temperature, keV (probably shouldn't change this without a better SigV function)
	SafetyFac: float = 3.5         # Plasma safety factor (q_edge)
	GamCD: float = 0.5             # Current drive efficiency, gamma_CD
	ElectEffCD: float = 0.5        # Electrical efficiency of CD system
	PowerRecirc: float = 0.05      # Fraction of extracted heat representing coolant pumping power
	ZEff: float = 1.00             # Plasma Zeff - note plasma dilution is not calculated, nor He checked! For radiation calculation


def simplesystemcode(inputs:InputParameters, print_out=True):

	# Physical inputs - don't alter
	WallRef = 0.6           # Wall reflectivity for synchrotron radiation
	SigV = 3.00e-22         # Velocity-averaged reactivity for fusion reaction at 15keV, m3 s-1
	NeutFastXC = 4.0        # Fast neutron slowing cross-section, barns
	NeutBreedXC = 950.0     # Slow neutron breeding cross-section, barns
	LiDens = 4.50e28        # Number density of Li atoms, m-3
	EFastNeut = 14.1        # Fast neutron energy, MeV
	EThNeut = 0.025         # Thermal neutron energy, eV
	Mu0 = 1.25664e-6        # Magnetic permeability, H m-1

	# Derived inputs, reverse-engineered from values in Freidberg
	EnRatioOne = 0.631654682 # En/(Ea+En+ELi)
	EnRatioTwo = 0.784      # (En+Ea)/(Ea+En+ELi)

	# Step 1: blanket and shield thickness
	NeutronMFP = 1.0/(LiDens * NeutFastXC * 1.0e-28)
	Li6Dens = inputs.Li6 * LiDens
	BreedLength = 1.0/(Li6Dens * NeutBreedXC * 1.0e-28)
	DelX = 2.0 * NeutronMFP * np.log(1.0 - 0.5*np.sqrt(1.0e6*EFastNeut/EThNeut)*(BreedLength/NeutronMFP)*np.log(1.0-inputs.NeutShield))
	BlnkThk = DelX + inputs.BlktSupport    # Total blanket thickness, m

	# Step 2: Plasma radius and coil thickness
	CoilZeta = (inputs.BMax**2)/(4.0 * Mu0 * inputs.SigmaMax * 1.0e6)
	VIoPe = (1.0/inputs.ThermalEff) * EnRatioOne * np.sqrt(inputs.Kappa) * (BlnkThk/inputs.WallLoad) * ((1.0+CoilZeta)/((1.0-np.sqrt(CoilZeta))**2))
	RMinor = ((1.0 + CoilZeta)/(2.0 * np.sqrt(CoilZeta)))*BlnkThk   # Plasma minor radius, m
	R0a = EnRatioOne * (inputs.GrossElecPower/inputs.WallLoad)
	R0b = 1.0/(inputs.ThermalEff*4.0*np.pi*np.pi*np.sqrt(inputs.Kappa))
	RMajor = R0a * R0b / RMinor     # Plasma major radius, m
	PlasVol = 2.0 * np.pi**2 * RMinor**2 * inputs.Kappa * RMajor   # Plasma Volume, m3

	# Others
	PressPrefix = 0.0000000000084 * np.sqrt(400)
	Pressure = PressPrefix * np.sqrt((inputs.PlasmaT**2)/(SigV * PlasVol))  # Plasma pressure, atm
	Aspect = RMajor/RMinor  # Plasma aspect ratio
	InvAspect = 1.0/Aspect # Inverse aspect ratio
	MagField = inputs.BMax * (RMajor - RMinor - BlnkThk)/RMajor   # Magnetic field in the plasma, T
	GeoFac = (1.17 - 0.65*InvAspect)/((1.0-InvAspect**2)**2)
	PlasCur = GeoFac * (5.0 * RMinor**2 * MagField)/(RMajor * inputs.SafetyFac) * (1.0 + inputs.Kappa**2)/2.0 # Assumes no triangularity for simplicity
	BPol = Mu0 * PlasCur * 1.0e6 / (2.0 * np.pi * RMinor * np.sqrt(inputs.Kappa))
	BetaPol = 2.0 * 100000.0 * Pressure * Mu0/(BPol**2)  # Plasma poloidal beta (normalised plasma pressure)
	BootFrac = min(0.5 * np.sqrt(InvAspect) * BetaPol, 1.0) # Bootstrap fraction, very simple formula
	DrivenCurrent = PlasCur * (1.0 - BootFrac)
	PlasmaDens = Pressure/(2.0*0.1602*inputs.PlasmaT)      # Plasma electron density, 10^20 m-3
	PowerCD = PlasmaDens * DrivenCurrent * RMajor / inputs.GamCD # MW
	PowerCDElec = PowerCD/inputs.ElectEffCD # Electrical power for current drive, MW
	ThermalPow = inputs.GrossElecPower / inputs.ThermalEff # Gross thermal power
	CoolantPower = ThermalPow * inputs.PowerRecirc # Power needed for coolant pumps
	NetElecPower = inputs.GrossElecPower - PowerCDElec - CoolantPower # Net electrical power once the other two are taken into account

	FusPDens = EnRatioTwo*inputs.GrossElecPower/(inputs.ThermalEff*PlasVol)   # FusionPowerDensity, MW m-3
	FusPower = FusPDens * PlasVol   # Fusion Power, MW
	PlasHeat = 0.2 * FusPower + PowerCD # Total plasma heating power, MW
	PTauE = 0.0562 * PlasCur**0.93 * MagField**0.15 * (10.0*PlasmaDens)**0.41 * PlasHeat**(-0.69) * RMajor**1.97 * inputs.Kappa**0.78 * InvAspect**0.58 * 2.5**0.19 # Prediction from IPB98(y,2), s
	ThermalE = Pressure * PlasVol / 10.0 # Thermal energy, MJ
	TauE = ThermalE/PlasHeat
	HFact = TauE/PTauE

	# Final parameters
	MagThk = (np.sqrt(CoilZeta)*(1.0+np.sqrt(CoilZeta)))/(1.0-np.sqrt(CoilZeta))*BlnkThk    # Magnet thickness, m
	Beta = 2.0 * 100000.0 * Pressure * Mu0/(MagField**2)  # Plasma beta (normalised plasma pressure)
	PlasSurf = 4.0 * np.pi**2 * RMinor * RMajor * np.sqrt(inputs.Kappa)  # Plasma surface area, m2
	WallLoadCalc = 0.8 * FusPower / PlasSurf  # Cross-check on neutron wall loading, MW m-2
	StableKappa = 1.46 + 0.5/(Aspect-1.0)  # Estimated maximum vertically-controllable kappa
	nG = PlasCur / (np.pi * RMinor**2) # Greenwald density limit

	# Radiation losses
	# Using forms from Matthews et al Nuc. Fus. 39 (1999)
	# Johner, Fus. Sci. Tech. 59 (2011)
	# ITER Physics Basics (1989)

	LineRad = ((inputs.ZEff-1.0) * PlasSurf**0.94 * PlasmaDens**1.8)/4.5  # Line radiation
	BremRad = 5.355e-3 * inputs.ZEff * PlasmaDens**2.0 * inputs.PlasmaT**0.5 * PlasVol  # Bremsstrahlung radiation
	Lambda0 = 77.7 * ((PlasmaDens * RMinor)/MagField)**0.5 # Lambda factor for synchrotron radiation
	GSyn = 0.16 * (inputs.PlasmaT/10.0)**1.5 * (1.0 + 5.7/(Aspect * (inputs.PlasmaT/10.0)**0.5))**0.5 # G factor for synchrotron radiation
	SynchRad = 6.2e-2 * (GSyn/Lambda0) * (1.0-WallRef)**0.5 * PlasmaDens * (inputs.PlasmaT/10.0) * MagField**2.0 * PlasVol # Synchrotron radiation
	DivLoad = (PlasHeat-LineRad-BremRad-SynchRad) * MagField / (Aspect * RMajor * inputs.SafetyFac) # Divertor heat load factor ~ 9.2 for EU-DEMO

	out_dict = {}
	# Set output dictionary
	out_dict["RMajor"] = RMajor
	out_dict["RMinor"] = RMinor
	out_dict["Aspect"] = Aspect
	out_dict["Kappa"] = inputs.Kappa
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
	out_dict["ZEff"] = inputs.ZEff # If this is too high the plasma is radiatively unstable, and highly diluted so fusion power will be depressed (although improvement of confinement with Zeff somewhat compensates for this)
	out_dict["MagThk"] = MagThk
	out_dict["BlnkThk"] = BlnkThk
	out_dict["Bore"] = RMajor - RMinor - BlnkThk - MagThk # Affects the size of cental solenoid and hence flux swing available
	out_dict["WallLoadCalc"] = WallLoadCalc

	# Set ExitCode parameter to 1 if computation yields nan or inf for any parameter in out_dict.
	ExitCode = 0
	for param in out_dict:
		if math.isnan(out_dict[param]) or math.isinf(out_dict[param]):
			ExitCode = 1

	out_dict['ExitCode'] = ExitCode

	# Print output
	if print_out:
		print("Simple systems code fusion power plant:\n")
		print('Major radius: {:2.2f} m'.format(RMajor))
		print('Minor radius: {:2.2f} m'.format(RMinor))
		print('Aspect ratio: {:2.2f}'.format(Aspect))
		print('Plasma elongation: {:2.2f}'.format(inputs.Kappa))
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
		print('Plasma ZEff: {:2.2f}'.format(inputs.ZEff))  # If this is too high the plasma is radiatively unstable, and highly diluted so fusion power will be depressed (although improvement of confinement with Zeff somewhat compensates for this)
		print("")
		print('Magnet thickness: {:2.2f} m'.format(MagThk))
		print('Blanket/shield thickness: {:2.2f} m'.format(BlnkThk))
		print('Bore: {:2.2f} m'.format(RMajor - RMinor - BlnkThk - MagThk))   # Affects the size of cental solenoid and hence flux swing available
		print('Wall load: {:2.2f} MW m-2'.format(WallLoadCalc))

	# Return input and output data as a complete design point
	design_point = dict(inputs.__dict__, **out_dict)
	return design_point