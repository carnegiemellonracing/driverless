import autograd.numpy as np
from scipy.io import loadmat

def Fx0(p, longslip, Fz, pressure, inclinagl):
    # Parameters
    ZETA1 = 1

    Fz0_ = p['FNOMIN'] * p['LFZO']
    dfz = (Fz - Fz0_) / Fz0_
    pi0 = p['NOMPRES']
    dpi = (pressure - pi0) / pi0
    Cx = p['PCX1'] * p['LCX']
    mux = p['LMUX'] * (p['PDX1'] + p['PDX2'] * dfz) * (1 + p['PPX3'] * dpi + p['PPX4'] * dpi**2) * (1 - p['PDX3'] * (inclinagl**2))
    Dx = mux * Fz * ZETA1
    SHx = p['LHX'] * (p['PHX1'] + p['PHX2'] * dfz)
    AMU = 10
    LMUX_ = AMU * p['LMUX'] / (1 + (AMU - 1) * p['LMUX'])
    SVx = ZETA1 * LMUX_ * p['LVX'] * Fz * (p['PVX1'] + p['PVX2'] * dfz)
    kappax = longslip + SHx
    kappaxSgn = np.sign(kappax)
    Ex = p['LEX'] * (p['PEX1'] + p['PEX2'] * dfz + p['PEX3'] * (dfz**2)) * (1 - p['PEX4'] * kappaxSgn)
    Kxk = p['LKX'] * Fz * (p['PKX1'] + p['PKX2'] * dfz) * np.exp(p['PKX3'] * dfz) * (1 + p['PPX1'] * dpi + p['PPX2'] * (dpi**2))
    Bx = Kxk / (Cx * Dx + np.finfo(float).eps)
    fx0 = Dx * np.sin(Cx * np.arctan(Bx * kappax - Ex * (Bx * kappax - np.arctan(Bx * kappax)))) + SVx
    return fx0, mux, Cx, Dx, Ex, dfz, Kxk


def Fx(p, longslip, slipangl, Fz, pressure, inclinagl):
    # Magic formula for Fx0
    fx0, mux, _, _, _, dfz, _ = Fx0(p, longslip, Fz, pressure, inclinagl)

    alphaAst = -np.tan(slipangl)
    gammaAst = np.sin(inclinagl)
    Cxa = p['RCX1']
    Exa = p['REX1'] + p['REX2'] * dfz
    SHxa = p['RHX1']
    alphaS = alphaAst + SHxa
    Bxa = p['LXAL'] * (p['RBX1'] + p['RBX3'] * (gammaAst**2)) * np.cos(np.arctan(p['RBX2'] * longslip))
    Gxa0 = np.cos(Cxa * np.arctan(Bxa * SHxa - Exa * (Bxa * SHxa - np.arctan(Bxa * SHxa))))
    Gxa = np.cos(Cxa * np.arctan(Bxa * alphaS - Exa * (Bxa * alphaS - np.arctan(Bxa * alphaS)))) / Gxa0
    fx = Gxa * fx0
    return fx

def Fy0(p, slipangl, Fz, pressure, inclinagl):
    # Parameters
    ZETA0 = 1
    ZETA2 = 1
    ZETA3 = 1
    ZETA4 = 1

    gammaAst = np.sin(inclinagl)
    gammaAst2 = gammaAst**2
    Fz0_ = p['FNOMIN'] * p['LFZO']
    dfz = (Fz - Fz0_) / Fz0_
    pi0 = p['NOMPRES']
    dpi = (pressure - pi0) / pi0
    dpi2 = dpi**2
    Cy = p['LCY'] * p['PCY1']
    muy = (p['PDY1'] + p['PDY2'] * dfz) * (1 + p['PPY3'] * dpi + p['PPY4'] * dpi2) * (1 - p['PDY3'] * gammaAst2) * p['LMUY']
    Dy = muy * Fz * ZETA2
    Kya = p['PKY1'] * Fz0_ * (1 + p['PPY1'] * dpi) * (1 - p['PKY3'] * np.abs(gammaAst)) * np.sin(p['PKY4'] * np.arctan(Fz / Fz0_ / ((p['PKY2'] + p['PKY5'] * gammaAst2) * (1 + p['PPY2'] * dpi)))) * p['LKY'] * ZETA3
    eps_kya = np.finfo(float).eps * np.where(Kya >= 0, 1, -1)
    # signKya = np.array(np.sign(Kya))
    # signKya[signKya == 0] = 1
    # Kya_ = Kya + np.finfo(float).eps * signKya
    Kya_ = Kya + eps_kya
    SVyg = ZETA2 * p['LKYC'] * p['LMUY'] * Fz * (p['PVY3'] + p['PVY4'] * dfz) * gammaAst
    Kyg0 = Fz * (p['PKY6'] + p['PKY7'] * dfz) * (1 + p['PPY5'] * dpi) * p['LKYC']
    SVy = ZETA2 * p['LMUY'] * p['LVY'] * Fz * (p['PVY1'] + p['PVY2'] * dfz) + SVyg
    SHy = p['LHY'] * (p['PHY1'] + p['PHY2'] * dfz) + (Kyg0 * gammaAst - SVyg) / Kya_ * ZETA0 + ZETA4 - 1
    alphay = slipangl + SHy
    alphaySgn = np.sign(slipangl)
    Ey = (p['PEY1'] + p['PEY2'] * dfz) * (1 + p['PEY5'] * gammaAst**2 - (p['PEY3'] + p['PEY4'] * gammaAst) * alphaySgn) * p['LEY']
    # signCy = np.sign(Cy)
    # signCy[signCy == 0] = 1
    eps_Cy = np.finfo(float).eps
    eps_Cy = eps_Cy * np.where(Cy >= 0, 1, -1)
    By = Kya / (Cy * Dy + eps_Cy)
    fy0 = Dy * np.sin(Cy * np.arctan(By * alphay - Ey * (By * alphay - np.arctan(By * alphay)))) + SVy
    return fy0, muy, dfz, Fz0_, dpi, By, Cy, Ey, Dy, Kya_, SVy, SHy

def Fy(p, longslip, slipangl, Fz, pressure, inclinagl):
    # Magic formula for Fy0
    fy0, muy, dfz, _, _, _, _, _, _, _, _, _ = Fy0(p, slipangl, Fz, pressure, inclinagl)
    
    gammaAst = np.sin(inclinagl)

    alphaAst = np.tan(slipangl)
    Byk = p['LYKA'] * (p['RBY1'] + p['RBY4'] * gammaAst**2) * np.cos(np.arctan(p['RBY2'] * (alphaAst - p['RBY3'])))
    Cyk = p['RCY1']
    DVyk = 1 * muy * Fz * (p['RVY1'] + p['RVY2'] * dfz + p['RVY3'] * gammaAst) * np.cos(np.arctan(p['RVY4'] * alphaAst))
    Eyk = p['REY1'] + p['REY2'] * dfz
    SHyk = p['RHY1'] + p['RHY2'] * dfz
    SVyk = p['LVYKA'] * DVyk * np.sin(p['RVY5'] * np.arctan(p['RVY6'] * longslip))
    kappaS = longslip + SHyk
    Gyk0 = np.cos(Cyk * np.arctan(Byk * SHyk - Eyk * (Byk * SHyk - np.arctan(Byk * SHyk))))
    Gyk = np.cos(Cyk * np.arctan(Byk * kappaS - Eyk * (Byk * kappaS - np.arctan(Byk * kappaS)))) / Gyk0
    Fy = Gyk * fy0 + SVyk
    return Fy


longitudinal_params_filename = "R20 18 inch long .mat"
lateral_params_filename = "R20 7-5 inch 10 psi .mat"

longitudinal_params = loadmat(longitudinal_params_filename)
lateral_params = loadmat(lateral_params_filename)

longitudinal_structured_array = longitudinal_params['mfparams'][0]
longitudinal_dict = {name: value.item() for name, value in zip(longitudinal_structured_array.dtype.names, longitudinal_structured_array[0]) if value.size == 1}
lateral_structured_array = lateral_params['mfparams'][0]
lateral_dict = {name: value.item() for name, value in zip(lateral_structured_array.dtype.names, lateral_structured_array[0]) if value.size == 1}


def get_forces(longslip, slipangl, Fz, pressure, inclinagl):
    fx = Fx(longitudinal_dict, longslip, slipangl, Fz, pressure, inclinagl)
    fy = Fy(lateral_dict, longslip, slipangl, Fz, pressure, inclinagl)
        # fy, _, _, _, _, _, _ = Fy(longitudinal_dict, longslip, slipangl, Fz, pressure, inclinagl)
    return fx, fy




# class Parameter:
#     def __init__(self, name, value=0):
#         self.name = name
#         self.value = value

# class ParameterFittable(Parameter):
#     def __init__(self, name, value=0, min_val=-float('inf'), max_val=float('inf'), fixed=False):
#         """
#         Initialize the ParameterFittable object.
        
#         Parameters:
#         - name (str): The name of the parameter.
#         - value (float): The current value of the parameter.
#         - min_val (float): The minimum value the optimizer will set.
#         - max_val (float): The maximum value the optimizer will set.
#         - fixed (bool): Whether the parameter is fixed (not optimized).
#         """
#         super().__init__(name, value)
#         self.min_val = min_val
#         self.max_val = max_val
#         self.fixed = fixed

#     def __repr__(self):
#         return f"ParameterFittable(name={self.name}, value={self.value}, min={self.min_val}, max={self.max_val}, fixed={self.fixed})"

# class TireParameters:
#     def __init__(self):
#         # Non-fittable parameters
#         self.FITTYP = Parameter("FITTYP", 61)
#         self.TYRESIDE = Parameter("TYRESIDE", 'LEFT')
#         self.LONGVL = Parameter("LONGVL", 10)
#         self.VXLOW = Parameter("VXLOW", 1)
#         self.ROAD_INCREMENT = Parameter("ROAD_INCREMENT")
#         self.ROAD_DIRECTION = Parameter("ROAD_DIRECTION")
        
#         self.UNLOADED_RADIUS = Parameter("UNLOADED_RADIUS")
#         self.WIDTH = Parameter("WIDTH")
#         self.RIM_RADIUS = Parameter("RIM_RADIUS")
#         self.RIM_WIDTH = Parameter("RIM_WIDTH")
#         self.ASPECT_RATIO = Parameter("ASPECT_RATIO")
        
#         self.INFLPRES = Parameter("INFLPRES")
#         self.NOMPRES = Parameter("NOMPRES", 1E5)

#         self.MASS = Parameter("MASS")
#         self.IXX = Parameter("IXX")
#         self.IYY = Parameter("IYY")
#         self.BELT_IXX = Parameter("BELT_IXX")
#         self.BELT_IYY = Parameter("BELT_IYY")
#         self.BELT_MASS = Parameter("BELT_MASS")
#         self.GRAVITY = Parameter("GRAVITY")
        
#         self.FNOMIN = Parameter("FNOMIN", 1500)
#         self.VERTICAL_DAMPING = Parameter("VERTICAL_DAMPING")
#         self.VERTICAL_STIFFNESS = Parameter("VERTICAL_STIFFNESS")
#         self.MC_CONTOUR_A = Parameter("MC_CONTOUR_A")
#         self.MC_CONTOUR_B = Parameter("MC_CONTOUR_B")
#         self.BREFF = Parameter("BREFF")
#         self.DREFF = Parameter("DREFF")
#         self.FREFF = Parameter("FREFF")
#         self.Q_RE0 = Parameter("Q_RE0")
#         self.Q_V1 = Parameter("Q_V1")
#         self.Q_V2 = Parameter("Q_V2")
#         self.Q_FZ2 = Parameter("Q_FZ2")
#         self.Q_FCX = Parameter("Q_FCX")
#         self.Q_FCY = Parameter("Q_FCY")
#         self.Q_FCY2 = Parameter("Q_FCY2")
#         self.Q_CAM = Parameter("Q_CAM")
#         self.Q_CAM1 = Parameter("Q_CAM1")
#         self.Q_CAM2 = Parameter("Q_CAM2")
#         self.Q_CAM3 = Parameter("Q_CAM3")
#         self.Q_FYS1 = Parameter("Q_FYS1")
#         self.Q_FYS2 = Parameter("Q_FYS2")
#         self.Q_FYS3 = Parameter("Q_FYS3")
#         self.PFZ1 = Parameter("PFZ1")
#         self.BOTTOM_OFFST = Parameter("BOTTOM_OFFST")
#         self.BOTTOM_STIFF = Parameter("BOTTOM_STIFF")
        
#         self.LONGITUDINAL_STIFFNESS = Parameter("LONGITUDINAL_STIFFNESS")
#         self.LATERAL_STIFFNESS = Parameter("LATERAL_STIFFNESS")
#         self.YAW_STIFFNESS = Parameter("YAW_STIFFNESS")
#         self.FREQ_LAT = Parameter("FREQ_LAT")
#         self.FREQ_LONG = Parameter("FREQ_LONG")
#         self.FREQ_WINDUP = Parameter("FREQ_WINDUP")
#         self.FREQ_YAW = Parameter("FREQ_YAW")
#         self.DAMP_LAT = Parameter("DAMP_LAT")
#         self.DAMP_LONG = Parameter("DAMP_LONG")
#         self.DAMP_RESIDUAL = Parameter("DAMP_RESIDUAL")
#         self.DAMP_VLOW = Parameter("DAMP_VLOW")
#         self.DAMP_WINDUP = Parameter("DAMP_WINDUP")
#         self.DAMP_YAW = Parameter("DAMP_YAW")
#         self.Q_BVX = Parameter("Q_BVX")
#         self.Q_BVT = Parameter("Q_BVT")
#         self.PCFX1 = Parameter("PCFX1")
#         self.PCFX2 = Parameter("PCFX2")
#         self.PCFX3 = Parameter("PCFX3")
#         self.PCFY1 = Parameter("PCFY1")
#         self.PCFY2 = Parameter("PCFY2")
#         self.PCFY3 = Parameter("PCFY3")
#         self.PCMZ1 = Parameter("PCMZ1")
        
#         self.Q_RA1 = Parameter("Q_RA1")
#         self.Q_RA2 = Parameter("Q_RA2")
#         self.Q_RB1 = Parameter("Q_RB1")
#         self.Q_RB2 = Parameter("Q_RB2")
#         self.ELLIPS_SHIFT = Parameter("ELLIPS_SHIFT")
#         self.ELLIPS_LENGTH = Parameter("ELLIPS_LENGTH")
#         self.ELLIPS_HEIGHT = Parameter("ELLIPS_HEIGHT")
#         self.ELLIPS_ORDER = Parameter("ELLIPS_ORDER")
#         self.ELLIPS_MAX_STEP = Parameter("ELLIPS_MAX_STEP")
#         self.ELLIPS_NWIDTH = Parameter("ELLIPS_NWIDTH")
#         self.ELLIPS_NLENGTH = Parameter("ELLIPS_NLENGTH")
#         self.ENV_C1 = Parameter("ENV_C1")
#         self.ENV_C2 = Parameter("ENV_C2")
#         self.Q_A1 = Parameter("Q_A1")
#         self.Q_A2 = Parameter("Q_A2")
        
#         self.PRESMIN = Parameter("PRESMIN")
#         self.PRESMAX = Parameter("PRESMAX")
        
#         self.FZMAX = Parameter("FZMAX")
#         self.FZMIN = Parameter("FZMIN")
        
#         self.KPUMIN = Parameter("KPUMIN")
#         self.KPUMAX = Parameter("KPUMAX")
        
#         self.ALPMIN = Parameter("ALPMIN")
#         self.ALPMAX = Parameter("ALPMAX")
        
#         self.CAMMIN = Parameter("CAMMIN")
#         self.CAMMAX = Parameter("CAMMAX")
        
#         # Scaling coefficients (fittable parameters)
#         self.LFZO = Parameter("LFZO", 1)
#         self.LCX = Parameter("LCX", 1)
#         self.LMUX = Parameter("LMUX", 1)
#         self.LEX = Parameter("LEX", 1)
#         self.LKX = Parameter("LKX", 1)
#         self.LHX = Parameter("LHX", 1)
#         self.LVX = Parameter("LVX", 1)
#         self.LCY = Parameter("LCY", 1)
#         self.LMUY = Parameter("LMUY", 1)
#         self.LEY = Parameter("LEY", 1)
#         self.LKY = Parameter("LKY", 1)
#         self.LKYC = Parameter("LKYC", 1)
#         self.LKZC = Parameter("LKZC", 1)
#         self.LHY = Parameter("LHY", 1)
#         self.LVY = Parameter("LVY", 1)
#         self.LTR = Parameter("LTR", 1)
#         self.LRES = Parameter("LRES", 1)
#         self.LXAL = Parameter("LXAL", 1)
#         self.LYKA = Parameter("LYKA", 1)
#         self.LVYKA = Parameter("LVYKA", 1)
#         self.LS = Parameter("LS", 1)
#         self.LMX = Parameter("LMX", 1)
#         self.LVMX = Parameter("LVMX", 1)
#         self.LMY = Parameter("LMY", 1)
#         self.LMP = Parameter("LMP", 1)
        
#         # Fittable longitudinal parameters
#         self.PCX1 = ParameterFittable("PCX1", 1.9, 1.5, float('inf'))
#         self.PDX1 = ParameterFittable("PDX1", 2, 1, 5)
#         self.PDX2 = ParameterFittable("PDX2", -0.01, -1, 0)
#         self.PDX3 = ParameterFittable("PDX3", 10, 0, 15)
#         self.PEX1 = ParameterFittable('PEX1', -1, -100, 0)
#         self.PEX2 = ParameterFittable('PEX2', 0.3, -3, 3)
#         self.PEX3 = ParameterFittable('PEX3', 0, -0.6, 0.6)
#         self.PEX4 = ParameterFittable('PEX4', 0, -0.9, 0.9)
#         self.PKX1 = ParameterFittable('PKX1', 20, 5, 100)
#         self.PKX2 = ParameterFittable('PKX2', 10, -30, 30)
#         self.PKX3 = ParameterFittable('PKX3', 0, -10, 10)
#         self.PHX1 = ParameterFittable('PHX1', 0, -0.1, 0.1)
#         self.PHX2 = ParameterFittable('PHX2', 0, -0.1, 0.1)
#         self.PVX1 = ParameterFittable('PVX1', 0, -0.1, 0.1)
#         self.PVX2 = ParameterFittable('PVX2', 0, -0.1, 0.1)
#         self.RBX1 = ParameterFittable('RBX1', 5, 5, 50)
#         self.RBX2 = ParameterFittable('RBX2', 5, 5, 50)
#         self.RBX3 = ParameterFittable('RBX3', 0, None, None)  # Without min and max values
#         self.RCX1 = ParameterFittable('RCX1', 1, -1, 1.2)
#         self.REX1 = ParameterFittable('REX1', -1, -10, 1)
#         self.REX2 = ParameterFittable('REX2', -0.1, -5, 1)
#         self.RHX1 = ParameterFittable('RHX1', 0, None, None)
#         self.PPX1 = ParameterFittable('PPX1', 0, None, None)
#         self.PPX2 = ParameterFittable('PPX2', 0, None, None)
#         self.PPX3 = ParameterFittable('PPX3', 0, None, None)
#         self.PPX4 = ParameterFittable('PPX4', 0, None, None)
#         # OVERTURNING_COEFFICIENTS
#         self.QSX1 = ParameterFittable('QSX1', 0, None, None)
#         self.QSX2 = ParameterFittable('QSX2', 1.5)
#         self.QSX3 = ParameterFittable('QSX3', 0.1)
#         self.QSX4 = ParameterFittable('QSX4', 0.1)
#         self.QSX5 = ParameterFittable('QSX5', 0, None, None)
#         self.QSX6 = ParameterFittable('QSX6', 0, None, None)
#         self.QSX7 = ParameterFittable('QSX7', 0.1)
#         self.QSX8 = ParameterFittable('QSX8', 0, None, None)
#         self.QSX9 = ParameterFittable('QSX9', 0, None, None)
#         self.QSX10 = ParameterFittable('QSX10', 0, None, None)
#         self.QSX11 = ParameterFittable('QSX11', 0, None, None)
#         self.QSX12 = ParameterFittable('QSX12', 0, -float('inf'), float('inf'), True)
#         self.QSX13 = ParameterFittable('QSX13', 0, -float('inf'), float('inf'), True)
#         self.QSX14 = ParameterFittable('QSX14', 0, -float('inf'), float('inf'), True)
#         self.PPMX1 = ParameterFittable('PPMX1', 0, None, None)
#         # LATERAL_COEFFICIENTS
#         self.PCY1 = ParameterFittable('PCY1', 2, 1.5, 5)
#         self.PDY1 = ParameterFittable('PDY1', 0.8, 0.1, 5)
#         self.PDY2 = ParameterFittable('PDY2', -0.05, -0.5, 0)
#         self.PDY3 = ParameterFittable('PDY3', 0, -10, 10)
#         self.PEY1 = ParameterFittable('PEY1', -0.8, -100, 100)
#         self.PEY2 = ParameterFittable('PEY2', -0.6, -100, 100)
#         self.PEY3 = ParameterFittable('PEY3', 0.1)
#         self.PEY4 = ParameterFittable('PEY4', 0, None, None)
#         self.PEY5 = ParameterFittable('PEY5', 0, None, None)
#         self.PKY1 = ParameterFittable('PKY1', -20, -100, -5)
#         self.PKY2 = ParameterFittable('PKY2', 2, -5, 5)
#         self.PKY3 = ParameterFittable('PKY3', 0, -1, 1)
#         self.PKY4 = ParameterFittable('PKY4', 2, 1.05, 2, True)
#         self.PKY5 = ParameterFittable('PKY5', 0, None, None)
#         self.PKY6 = ParameterFittable('PKY6', 0, None, None)
#         self.PKY7 = ParameterFittable('PKY7', 0, None, None)
#         self.PHY1 = ParameterFittable('PHY1', 0, None, None)
#         self.PHY2 = ParameterFittable('PHY2', 0, None, None)
#         self.PVY1 = ParameterFittable('PVY1', 0, None, None)
#         self.PVY2 = ParameterFittable('PVY2', 0, None, None)
#         self.PVY3 = ParameterFittable('PVY3', 0, None, None)
#         self.PVY4 = ParameterFittable('PVY4', 0, None, None)
#         self.RBY1 = ParameterFittable('RBY1', 5, 2, 40)
#         self.RBY2 = ParameterFittable('RBY2', 2, 2, 40)
#         self.RBY3 = ParameterFittable('RBY3', 0.02, -0.15, 0.15)
#         self.RBY4 = ParameterFittable('RBY4', 0, -90, 90)
#         self.RCY1 = ParameterFittable('RCY1', 1, 0.8, 5)
#         self.REY1 = ParameterFittable('REY1', -0.1, -6, 1)
#         self.REY2 = ParameterFittable('REY2', 0.1, -1, 5)
#         self.RHY1 = ParameterFittable('RHY1', 0, None, None)
#         self.RHY2 = ParameterFittable('RHY2', 0, None, None)
#         self.RVY1 = ParameterFittable('RVY1', 0, None, None)
#         self.RVY2 = ParameterFittable('RVY2', 0, None, None)
#         self.RVY3 = ParameterFittable('RVY3', 0, None, None)
#         self.RVY4 = ParameterFittable('RVY4', 0, None, None)
#         self.RVY5 = ParameterFittable('RVY5', 0, None, None)
#         self.RVY6 = ParameterFittable('RVY6', 0, None, None)
#         self.PPY1 = ParameterFittable('PPY1', 0.1, -2, 2)
#         self.PPY2 = ParameterFittable('PPY2', 0.1, -2, 2)
#         self.PPY3 = ParameterFittable('PPY3', 0, -2, 1)
#         self.PPY4 = ParameterFittable('PPY4', 0, -2, 1)
#         self.PPY5 = ParameterFittable('PPY5', 0, None, None)
#         # ROLLING_COEFFICIENTS
#         self.QSY1 = ParameterFittable('QSY1', 0, None, None)
#         self.QSY2 = ParameterFittable('QSY2', 0, None, None)
#         self.QSY3 = ParameterFittable('QSY3', 0, None, None)
#         self.QSY4 = ParameterFittable('QSY4', 0, None, None)
#         self.QSY5 = ParameterFittable('QSY5', 0, None, None)
#         self.QSY6 = ParameterFittable('QSY6', 0, None, None)
#         self.QSY7 = ParameterFittable('QSY7', 0, None, None)
#         self.QSY8 = ParameterFittable('QSY8', 0, None, None)
