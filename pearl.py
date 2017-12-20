import numpy as np
import matplotlib.pyplot as plt
import re
import datetime as dt
import socket

hostname = socket.gethostname()
if hostname == 'mbp62':
    Now = '/home/lve/pearl/Data/pearl_current.dat'
    now = '/home/lve/pearl/data/pearl_current.dat'
    conf = '/home/lve/pearl/COMMISSIONING/'
    tmpdir = '/home/lve/pearl/tmp/'
elif hostname == 'll':
    Now = '/home/lve/pearl/Data/pearl_current.dat'
    now = '/home/lve/pearl/data/pearl_current.dat'
    conf = '/home/lve/pearl/COMMISSIONING/'
    tmpdir = '/home/lve/pearl/tmp/'
elif hostname == 'pearl':
    now = '/var/www/owncloud/data/pearl/files/Data/pearl_current.dat'
    conf = '/var/www/owncloud/data/pearl/files/COMMISSIONING/'
    tmpdir = '/var/www/owncloud/data/pearl/files/tmp/'
elif hostname == 'TUD277455':
    now = 'M:/tnw/rrrrid/rid/F&S/PEARL/Data/pearl_current.dat'
    conf = 'D:/PEARL/COMMISSIONING/'
    tmpdir = 'D:/PEARL/Data/tmp/'

can6mm = conf+'background/empty_can_6mm.txt'
#deteff = np.load(conf+'deteff_mar2016.npy')
#DETEFF = np.load(conf+'assembled_vanadium_nospurions_perspexed.npy')
#DETEFF = np.load(conf+'vanadium_perspex_Jul2016.npy')
DETEFF533 = np.load(conf+'vanadium/vanadium_perspex_nov2016.npy').flatten()
DETEFF = DETEFF533
DETEFF133 = DETEFF533
DETEFF733 = DETEFF533
DETEFF955 = np.load(conf+'vanadium/vanadium_955_mar2017.npy').flatten()
DETEFF755 = np.load(conf+'vanadium/vanadium_755_mar2017.npy').flatten()
#tth = np.load(conf+'tth_mar2016.npy')
tth = np.load(conf+'tth_dec2016.npy')

vartype2016={}
vartype2016['ScPoint']='i4'
vartype2016['Sc Point']='i4'
vartype2016['OxfordSetpoint']='f4'
vartype2016['ITC']='f4'
vartype2016['ITC_WAIT_TIME']='f4'
vartype2016['OXITCVALUE1']='f4'
vartype2016['OXITCPB'] = 'f4'
vartype2016['OXITCIT'] = 'f4'
vartype2016['OXITCDT'] = 'f4'
vartype2016['OXITCPOW'] = 'f4'
vartype2016['oxford_temperaturetolerance_K'] = 'f4'
vartype2016['WAVELENGTH']='f4'
vartype2016['Mon']='f4'
vartype2016['Detsum']='f4'
vartype2016['StartT']='S19'
vartype2016['MeasT']='f4'
vartype2016['DeltaT']='f4'
vartype2016['USMCPOSA']='f4'
vartype2016['XangleSet']='f4'


vartype2017={}
vartype2017['measurement_nr']='i4'
vartype2017['oxford_itc502_temperature1_set_K']='f4'
vartype2017['oxford_itc502_temperature1_K']='f4'
vartype2017['oxford_itc502_P_set'] = 'f4'
vartype2017['oxford_itc502_I_set'] = 'f4'
vartype2017['oxford_itc502_D_set'] = 'f4'
vartype2017['oxford_itc502_heater_voltage_%'] = 'f4'
vartype2017['oxford_temperature_tolerance_K'] = 'f4'
vartype2017['oxford_itc502_wait_time_sec'] = 'f4'
vartype2017['monochromator_wavelength_AA']='f4'
vartype2017['monochromator_angle_deg']='f4'
vartype2017['monochromator_angle_set_deg']='f4'
vartype2017['monitor_counts']='f4'
vartype2017['detector_sum_counts']='f4'
vartype2017['start_time_yyyy-mm-dd_hh:mm:ss']='S19'
vartype2017['measuring_time_sec']='f4'
vartype2017['voltcraft_powersupply_voltage_V']='f4'
vartype2017['voltcraft_powersupply_current_A']='f4'
vartype2017['voltcraft_actual_current_A']='f4'
vartype2017['voltcraft_actual_voltage_V']='f4'
vartype2017['pt100_temperature1_degC']='f4'
vartype2017['pt100_temperature2_degC']='f4'

vartype = vartype2017

## graphics
#def crosshair():
#    """ def crosshair() """
#    from matplotlib.widgets import Cursor
#    cursor = Cursor(plt.gca(), useblit=True, color='grey', linewidth=1)

def mask_lines(*items):
    for i in items:
        if type(i) is not int:
            print('all items in pearl.mask_lines(items) should be integers')
            return
    lines = plt.gca().get_lines()
    if len(items) is 0:
        n=1
        for l in lines:
            print('%i %s'%(n,l))
        print('this list starts at 1; enter 0 to make all lines visible')
    else:
        if len(items) is 0 and items is 0:
            for l in lines:
                lines[l].set_visible(True)
        else:
            for i in items:
                if lines[i-1].get_visible():
                    lines[i-1].set_visible(False)
                else:
                    lines[i-1].set_visible(True)


Lorentz = 1 / ( 2 * ( np.sin(tth/360*np.pi) )**2 * np.cos(tth/360*np.pi) )
d_Al = 4.0495e-10
d_Ge = 5.6575e-10
d_Ni = 3.5238e-10
d_V = 3.024e-10
d_Fe = 2.8665e-10
d_Cu = 3.6149e-10
d_Cr = 2.91e-10
d_Mo = 3.147e-10
d_W = 3.1652e-10
d_Li = 3.51e-10
d_Au = 4.0495e-10
d_Ag = 4.0853e-10
d_He = 4.242e-10
d_Na = 4.2906e-10
d_Pb = 4.9508e-10
d_B = 5.06e-10
d_Th = 5.0842e-10
d_Ar = 5.256e-10
d_K = 5.328e-10
d_Si = 5.4309e-10
d_Ca = 5.5884e-10
d_Kr = 5.706e-10
d_Cs = 6.141e-10
d_Xe  =6.2023e-10
d_Mn = 8.9125e-10


ab_Si2O3 = 4.759355
c_Si2O3 = 12.99231

mono_theta= 75.0/180.0*np.pi

d_400 = d_Ge/np.sqrt(16)
lambda_400 = 2*d_400*np.sin(mono_theta)
lambda_400_AA = lambda_400 * 1e10
d_800 = d_Ge/np.sqrt(64)
lambda_800 = 2*d_800*np.sin(mono_theta)
lambda_800_AA = lambda_800 * 1e10
d_311 = d_Ge/np.sqrt(9+2)
lambda_311 = 2*d_311*np.sin(mono_theta)
lambda_311_AA = lambda_311 * 1e10
d_511 = d_Ge/np.sqrt(25+2)
lambda_511 = 2*d_511*np.sin(mono_theta)
lambda_511_AA = lambda_511 * 1e10
d_711 = d_Ge/np.sqrt(49+2)
lambda_711 = 2*d_711*np.sin(mono_theta)
lambda_711_AA = lambda_711 * 1e10
d_911 = d_Ge/np.sqrt(81+2)
lambda_911 = 2*d_911*np.sin(mono_theta)
lambda_911_AA = lambda_911 * 1e10
d_533 = d_Ge/np.sqrt(25+2*9)
lambda_533 = 2*d_533*np.sin(mono_theta)
lambda_533_AA = lambda_533 * 1e10
d_733 = d_Ge/np.sqrt(49+2*9)
lambda_733 = 2*d_733*np.sin(mono_theta)
lambda_733_AA = lambda_733 * 1e10
d_133 = d_Ge/np.sqrt(1+2*9)
lambda_133 = 2*d_133*np.sin(mono_theta)
lambda_133_AA = lambda_133 * 1e10
d_422 = d_Ge/np.sqrt(4+4+16)
lambda_422 = 2*d_422*np.sin(mono_theta)
lambda_422_AA = lambda_422 * 1e10
d_044 = d_Ge/np.sqrt(0+16+16)
lambda_044 = 2*d_044*np.sin(mono_theta)
lambda_044_AA = lambda_044 * 1e10
d_155 = d_Ge/np.sqrt(1+2*25)
lambda_155 = 2*d_155*np.sin(mono_theta)
lambda_155_AA = lambda_155 * 1e10
d_355 = d_Ge/np.sqrt(9+2*25)
lambda_355 = 2*d_355*np.sin(mono_theta)
lambda_355_AA = lambda_355 * 1e10
d_755 = d_Ge/np.sqrt(49+2*25)
lambda_755 = 2*d_755*np.sin(mono_theta)
lambda_755_AA = lambda_755 * 1e10
d_955 = d_Ge/np.sqrt(81+2*25)
lambda_955 = 2*d_955*np.sin(mono_theta)
lambda_955_AA = lambda_955 * 1e10
d_111 = d_Ge/np.sqrt(1+1+1)
lambda_111 = 2*d_111*np.sin(mono_theta)
lambda_111_AA = lambda_111 * 1e10
d_333 = d_Ge/np.sqrt(9+9+9)
lambda_333 = 2*d_333*np.sin(mono_theta)
lambda_333_AA = lambda_333 * 1e10
d_555 = d_Ge/np.sqrt(75)
lambda_555 = 2*d_555*np.sin(mono_theta)
lambda_555_AA = lambda_555 * 1e10

# gsas-2 fitted parameters:
lambda_133_cal_AA = 2.50849008034 # nov 2017
lambda_533_cal_AA = 1.66713367184 # nov 2017
lambda_733_cal_AA = 1.33555024934 # nov 2017
lambda_755_cal_AA = 1.09878762126
lambda_955_cal_AA =0.955073316184

zero_133_cal_deg = -0.269626673459 # nov 2017
zero_533_cal_deg = -0.190805585383 # nov 2017
zero_733_cal_deg = -0.30676897362 # nov 2017

#lambda_533 = 2*d_533*np.sin(mono_theta)
#lambda_733 = 2*d_733*np.sin(mono_theta)
#lambda_755 = 2*d_755*np.sin(mono_theta)
#lambda_311 = 2*d_311*np.sin(mono_theta)
#lambda_133 = 2*d_133*np.sin(mono_theta)
#lambda_422 = 2*d_422*np.sin(mono_theta)
#lambda_533_AA = lambda_533 *1e10
#lambda_733_AA = lambda_733 *1e10
#lambda_755_AA = lambda_755 *1e10
#lambda_311_AA = lambda_311 *1e10
#lambda_133_AA = lambda_133 *1e10
#lambda_422_AA = lambda_422 *1e10
L1 = 6.750
L2 = 2.000
L3 = 1.145
source_width = 0.16 # full width /diameter of tube
mono_width = 0.050 # 50mm
sample_diameter = 0.010 # 10mm 
pixel_width = 0.0021 # 2.1mm
pixel_width_deg = pixel_width / (2*np.pi*1145) *360
A1 = np.arctan( (mono_width/2 + source_width/2) / L1 ) /np.pi*180
A2 = np.arctan( (mono_width/2 + sample_diameter/2) / L2 ) /np.pi*180
A3 = np.arctan( (sample_diameter/2 + pixel_width/2) / L3 )/np.pi*180
beta = 0.45 # average over some crystals
mono_crystals_y = np.asarray([15.49, 80.42, 145.39])/1000
mono_R = 2 / ( (1/L1 + 1/L2)*np.sin(mono_theta) )
mono_crystal_thickness = 0.0132
#mono_total_thickness = 0.019 #crystal+alu_backplate+sapphire-sapphire_wells
#mono_sapphire_R = mono_R -1/3*mono_crystal_thickness+mono_total_thickness
#mono_crystals_tilt_alpha= np.arcsin(mono_crystals_y/mono_sapphire_R)
#mono_marker_distance = np.asarray([0.008,0.004]) #vertical inside instrument
#mono_crystal_tilt_marker_1_2_mm = mono_crystals_y *mono_marker_distance[0] /mono_sapphire_R *1000
#mono_crystal_tilt_marker_1_3_mm = mono_crystals_y *mono_marker_distance[1] /mono_sapphire_R *1000
S_over_L3 = 0.023923444976076555
H_over_L3 = 0.09569377990430622

FCC = {'111': np.sqrt(3), '200': 2, '220': np.sqrt(8), '311': np.sqrt(11), '222': np.sqrt(12), '400': np.sqrt(16), '331': np.sqrt(19), '420': np.sqrt(20), '422': np.sqrt(24), '333': np.sqrt(27)}
BCC = {'110': np.sqrt(2), '200': 2, '211': np.sqrt(6), '220': np.sqrt(8), '310': np.sqrt(10), '222': np.sqrt(12), '400': np.sqrt(16), '330': np.sqrt(18), '420': np.sqrt(20), '422': np.sqrt(24), '510': np.sqrt(26)}

class xrdml(object):
    def __init__(self, filename, label = None, color = None, marker = None, cps = True, plot=True, x=None):
        import xrdtools
        self.content = xrdtools.read_xrdml(filename)
        self.twotheta = self.content['2Theta']
        self.x = self.content['x']
        self.y = self.content['data']
        self.wavelength = self.content['Lambda']
        self.measuringtime_per_step = self.content['time']
        self.y_cps = self.y / self.measuringtime_per_step
        self.Q = 4 * np.pi / self.wavelength * np.sin( self.x / 360.0 * np.pi )
        self.d = 2 * np.pi / self.Q
        if cps == True:
            self.cps = True
            print('cps is true')
            self.Y = self.y_cps
        else:
            self.cps = False
            print('cps is false')
            self.Y = self.y
        if x is 'Q':
            self.x = self.Q
        if x is 'd':
            self.x = self.d
        if label is not None:
            self.label = label
        else:
            self.label = self.content['filename']
        if plot == True:
            self.plot()

    def plot(self, label=None, color=None, marker=None, x=None, cps=True):
        """ def plot(self, label=None, marker=None, cps=None, color=None, x=None):"""
        if label:
            self.label = label
        xlabel='2$\Theta$ /deg'
        if x is 'Q':
            self.x=self.Q
            xlabel='Q /$\\AA^{-1}$'
            print('setting Q as x-axis')
        elif x is 'd':
            self.x=self.d
            xlabel='d-spacing /$\\AA$'
        if marker:
            self.marker = marker
        if marker:
            self.marker = marker
        if cps is True:
            self.cps = cps
            self.Y = self.y_cps
        if cps is False:
            self.Y = self.content['data']
        if color:
            self.plotcolor = color
        if self.cps:
            plt.ylabel('counts per second')
        else:
            plt.ylabel('counts')
        plt.plot(self.x, self.Y, markersize=3, label=self.label)
        plt.legend()
        plt.grid()


def reflection_to_hkl(reflection):
    """ def reflection_to_hkl(111) """
    reflection = int(reflection)
    if reflection >999:
        print('--- you cannot have more than 999 for the reflection.')
    else:
        h = np.int( (reflection - reflection %100)/100 )
        k = np.int( (reflection %100 - reflection %10)/10 )
        l = np.int( reflection %10 -reflection %1 )
        return h,k,l

def d_to_tth(d,mono_setting):
    """ d_to_tth(d, mono_setting)
        d_to_tth(1e-10, 533)
        """
    hmono, kmono, lmono = reflection_to_hkl(mono_setting)
    wavelength = 2 * Ge(533).d * np.sin(mono_theta)
    return np.arcsin (wavelength / (2 * d) ) *360/np.pi

tth533_Al111 = 360/np.pi * np.arcsin( lambda_533 /2 / (d_Al / np.sqrt(3) ) )
tth533_Al200 = 360/np.pi * np.arcsin( lambda_533 /2 / (d_Al / np.sqrt(4) ) )
tth533_Al113 = 360/np.pi * np.arcsin( lambda_533 /2 / (d_Al / np.sqrt(11) ) )
tth533_Al222 = 360/np.pi * np.arcsin( lambda_533 /2 / (d_Al / np.sqrt(12) ) )
tth533_Al400 = 360/np.pi * np.arcsin( lambda_533 /2 / (d_Al / np.sqrt(16) ) )
tth533_Al133 = 360/np.pi * np.arcsin( lambda_533 /2 / (d_Al / np.sqrt(19) ) )

detector_nr_o_pixels=1408

def angle_xray_to_pearl(xray_twotheta,pearl_lambda = 1.667,xray_lambda=1.54):
    """ def angle_xray_to_pearl( xray_twotheta, pearl_lambda = 1.667, xray_lambda=1.54) """
    pearl_twotheta = 0
    pearl_twotheta = 360.0/np.pi * np.arcsin(  pearl_lambda / xray_lambda * np.sin(xray_twotheta/360.0*np.pi)  )
    print('xray lambda = %3.2f \n xray_twotheta = %3.2f \n pearl lambda = %3.2f \n pearl twotheta = %3.2f deg'% (xray_lambda, xray_twotheta, pearl_lambda, pearl_twotheta))
    return(pearl_twotheta)

def angle_533_to_133(tth_533):
    """ def angle_533_to_133(tth_533) """
    tth_133 = 0
    tth_133 = 360.0/np.pi * np.arcsin(  lambda_133 / lambda_533 * np.sin(tth_533/360.0*np.pi)  )
    return(tth_133)

def angle_533_to_733(tth_533):
    """ def angle_533_to_733(tth_533) """
    tth_733 = 0
    tth_733 = 360.0/np.pi * np.arcsin(  lambda_733 / lambda_533 * np.sin(tth_533/360.0*np.pi)  )
    return(tth_733)

def angle_133_to_533(tth_133):
    """ def angle_133_to_533(tth_133) """
    tth_533 = 0
    tth_533 = 360.0/np.pi * np.arcsin(  lambda_533 / lambda_133 * np.sin(tth_133/360.0*np.pi)  )
    return(tth_533)

def angle_733_to_533(tth_733):
    """ def angle_733_to_533(tth_733) """
    tth_533 = 0
    tth_533 = 360.0/np.pi * np.arcsin(  lambda_533 / lambda_733 * np.sin(tth_733/360.0*np.pi)  )
    return(tth_533)

def angle_133_to_733(tth_133):
    """ def angle_133_to_733(tth_133) """
    tth_733 = 0
    tth_733 = 360.0/np.pi * np.arcsin(  lambda_733 / lambda_133 * np.sin(tth_133/360.0*np.pi)  )
    return(tth_733)

def angle_733_to_133(tth_733):
    """ def angle_733_to_133(tth_733) """
    tth_133 = 0
    tth_133 = 360.0/np.pi * np.arcsin(  lambda_133 / lambda_733 * np.sin(tth_733/360.0*np.pi)  )
    return(tth_133)

def fit_gaussian_bkg(x,y):
    """ pars = fit_gaussian_bkg(x,y)
    pars = (mu0,amp,sigma,bkg) """
    pars0 = guess_pars_gaussian_bkg(x,y)
    from scipy.optimize import curve_fit
    plt.plot(x, gaussian_bkg(x, *pars0),':')
    pars, pconv = curve_fit( gaussian_bkg, x, y, pars0 )
    plt.plot(x, gaussian_bkg(x, *pars),'--')
    return(pars)

def gaussian_bkg(x, *p):
    """ def gaussian_bkg(x, *p) 
    mu0 = p[0]
    amp = p[1]
    sigma = p[2]
    bkg = p[3]
    """
    mu0 = p[0]
    amp = p[1]
    sigma = p[2]
    bkg = p[3]
    return np.array( bkg + amp / (sigma * np.sqrt(2*np.pi)) * np.exp( -(x-mu0)**2 / 2 / sigma**2 ) )

def guess_pars_gaussian_bkg(x,y):
    """ def guess_pars_gaussian_bkg(x,y) """ 
    bkg = y.min()
    amp = y.max() - bkg
    mu0 = x[ y.argmax() ]
    yhm = amp/2 + bkg
    xtop = x[ y > yhm ]
    fwhm = xtop[-1] - xtop[0]
    sigma = fwhm / np.sqrt(2*np.log(2))
    return (mu0, amp, sigma, bkg)

class gsas_csv(object):
    def __init__(self,filename, label = None, color = None, marker = None):
        import numpy as np
        import matplotlib.pylab as plt
        a=np.genfromtxt(filename, skip_header=5, delimiter=',', usecols=(1,2,3,4,5))
        self.x = a[:,0]
        self.tth = self.x
        self.y_obs = a[:,1]
        self.y_calc = a[:,2]
        self.y_bkg = a[:,3]
        self.y_diff = a[:,4]
        plt.plot(self.x, self.y_obs,'o', label='obs', markersize=2)
        plt.plot(self.x, self.y_calc, label='calc')
        plt.plot(self.x, self.y_bkg, label='bkg')
        plt.plot(self.x, self.y_diff, label='diff')
        plt.legend()
        plt.xlabel('2$\Theta$ /deg')
        plt.ylabel('counts')

class fp_prf(object):
    def __init__(self,filename, label = None, color = None, marker = None):
        import numpy as np
        import matplotlib.pylab as plt
        dummy = open(filename).readlines()
        #line 1
        self.label = dummy[0]
        #line 2
        self.Nphases, self.Ntth, self.wavelength, self.wavelength2, self.zero, self.par1, self.par2, self.par3 = dummy[1].split()
        self.Nphases = int(self.Nphases)
        self.Ntth = int(self.Ntth)
        self.wavelength = float(self.wavelength)
        self.wavelength2 = float(self.wavelength2)
        self.zero = float(self.zero)
        self.par1 = float(self.par1)
        self.par2 = float(self.par2)
        self.par3 = int(self.par3)
        #line 3
        npars=self.Nphases*2+1
        items=np.zeros(npars, dtype='int')
        items=dummy[2].split()
        self.Nrefl=np.zeros(self.Nphases,dtype='int')
        self.Npropvectors=np.zeros(self.Nphases,dtype='int')
        for nph in range(self.Nphases):
            self.Nrefl[nph] = items.pop(0)
        for nph in range(self.Nphases):
            self.Npropvectors[nph] = items.pop(0)
        self.Nexcludes = int(items.pop(0))
        self.excludelimits = np.zeros([self.Nexcludes,2])
        for excluderange in np.arange(self.Nexcludes):
            self.excludelimits[excluderange,0] = float(dummy[3+excluderange].split()[0])
            self.excludelimits[excluderange,1] = float(dummy[3+excluderange].split()[1])
        self.labels = dummy[4+self.Nexcludes].split('\t')
        ncols=len(dummy[5+self.Nexcludes].split('\t'))
        if ncols == 5: # the tickmarks are below the data block
            self.prfFileType = -3 
            self.data = np.genfromtxt(filename, skip_header = 5+self.Nexcludes, delimiter='\t', skip_footer=np.sum(self.Nrefl))
            self.x = self.data[:,0] ; self.tth = self.x
            self.y_obs = self.data[:,1]
            self.y_calc = self.data[:,2]
            self.y_diff = self.data[:,3]
            self.y_bkg = self.data[:,4]
            self.y_bkg = self.data[:,4]
            plt.plot(self.x, self.y_obs,'o', label='obs', markersize=2)
            plt.plot(self.x, self.y_calc, label='calc')
            plt.plot(self.x, self.y_bkg, label='bkg')
            plt.plot(self.x, self.y_diff, label='diff')
            dy = plt.gca().get_ylim()
            dy = (dy[1]-dy[0])/100 # tickmark height of peakpos
            # read reflection labels
            startline = 4+self.Nexcludes+self.Ntth
            self.reflections = dict()
            for R in np.arange(np.sum(self.Nrefl)):
                Label=dummy[startline+R].split('\t')[6]
                peakposx=float(dummy[startline+R].split('\t')[0])
                peakposy=float(dummy[startline+R].split('\t')[5])
                self.reflections[ Label ] = peakposx
                plt.plot([peakposx,peakposx],[peakposy,peakposy+dy],color='grey')
                plt.text(peakposx,peakposy-dy, Label, fontsize=8, rotation=90, verticalalignment='top', horizontalalignment='center')
        elif ncols == 9: # the tickmarks are behind (right of) the data block
            self.prfFileType = 3
            self.skip = 4 + self.Nexcludes
            ntth = len(dummy) - self.skip
            self.data = np.zeros( (ntth, 5) )
            for i in np.arange(ntth):
                self.data[i,:] = dummy[i+self.skip].split()[0:5] 
            self.x = self.data[:,0] ; self.tth = self.x
            self.y_obs = self.data[:,1]
            self.y_calc = self.data[:,2]
            self.y_diff = self.data[:,3]
            self.y_bkg = self.data[:,4]
            self.y_bkg = self.data[:,4]
            plt.plot(self.x, self.y_obs,'o', label='obs', markersize=2)
            plt.plot(self.x, self.y_calc, label='calc')
            plt.plot(self.x, self.y_bkg, label='bkg')
            plt.plot(self.x, self.y_diff, label='diff')
            dy = plt.gca().get_ylim()
            dy = (dy[1]-dy[0])/100 # tickmark height of peakpos
            # read reflection labels
            self.reflections = dict()
            for R in np.arange(np.sum(self.Nrefl)):
                peakposx, peakposy, Label, phase =dummy[self.skip+R].split('\t')[5:]
                peakposx = float(peakposx)
                peakposy = float(peakposy)
                # phase = int(phase) # a 2 digit string like "0  1"
                self.reflections[ Label ] = peakposx
                plt.plot([peakposx,peakposx],[peakposy,peakposy+dy],color='grey')
                plt.text(peakposx,peakposy-dy, Label, fontsize=8, rotation=90, verticalalignment='top', horizontalalignment='center')
        else: 
            print('ncols not 5, nor 9: do not understand file format of PRF file')
        plt.legend()
        plt.xlabel('2$\Theta$ /deg')
        plt.ylabel('counts')


def readpearl(filename):
	header={}
	a=[lines for lines in open(filename).readlines()]
	header[a[0].split('\t')[0]]=a[0].split('\t')[1]
	for i in range(int(header['npar'])):
		header[a[i+1].split('\t')[0]]=a[i+1].split('\t')[1]
	for i in range(len(a[18].split('\t'))):
		header[a[18].split('\t')[i]] = a[20].split('\t')[i]
	data = np.genfromtxt(filename, skip_header=23)
	return [header,data]
	
def oldread_file(filename):
    """ This reads the 'old' file format """
    dummy=np.genfromtxt(filename)
    title=filename.split('.dat')[0]
    x = dummy[0:-1,0]
    y = dummy[0:-1,1]
    measuring_time_sec = dummy[-1,1]
    y = y.repeat(1).reshape(len(y),1) # adapt to generic y format
    return(x,y,measuring_time_sec,title)

def currentread_file(filename):
    if filename == None:
        filename = now
    a=[lines for lines in open(filename).readlines()]
    currentfilename = ''.join(a[0].split('\t')[1])
    title = ''.join(a[1].split('\t')[1:-1])
    sample = ''.join(a[2].split('\t')[1:-1])
    comment = ''.join(a[3].split('\t')[1:-1])
    measuring_time_sec = int(a[4].split('\t')[1])
    wavelength = ''.join(a[5].split('\t')[1:-1])
    data = np.genfromtxt(filename, skip_header=6)[:,1:3]
    x = data[:,0]
    y = data[:,1]
    npoints = 1
    y = y.repeat(1).reshape(len(y),npoints) # adapt to generic y format
    label = '( CURRENT: ' + currentfilename + ') ' + title + ' ' + sample + ' ' + comment
    header = {}
    header['file_format_version'] = 'pearlcurrent'
    header['file_name'] = currentfilename
    header['measurement_sample_name'] = sample
    header['measurement_type'] = 'single'
    header['monochromator_wavelength_AA'] = wavelength
    header['measurement_nr_o_points'] = 1
    header['experiment_name'] = title
    header['experiment_user_name'] = ''
    header['label'] = label
    pars = np.empty(1,dtype=[('measurement_nr','i4'), ('monochromator_wavelength_AA','f4'), ('monitor_counts','f4'), ('detector_sum_counts','f4'), ('start_time_yyyy-mm-dd_hh:mm:ss','S19'), ('measuring_time_sec','f4')])
    pars[0] = ( 0, wavelength, 0, y.sum(), '', measuring_time_sec )
    return x, y, currentfilename, label, title, sample, comment, measuring_time_sec, wavelength, npoints, header, pars

def getpars_read_2016file(stringarray,npoints,file_format_table_constants_nr_o_rows,ncolumscpts):
    print('getpars_read_2016file():')
    for n in range(npoints):
        print('n, npoints = {},{}'.format(n,npoints))
        print('n, npoints = {},{}'.format(n,npoints))
        print(stringarray[file_format_table_constants_nr_o_rows+2+n])
        print(stringarray[file_format_table_constants_nr_o_rows+2+n].split('\t')[0:ncolumscpts])

def read_2016file(*args):
    """ This reads the 'new' file format """
    print('running read_2016file()')
    file_format_version = ''
    measurement_type=''
    if not len(args)>1:
        print('need filename, file_format_version, measurement_type')
        return
    def convert_parnames2016(parnames):
        vartype={}
        if 'ScPoint' in parnames:
            parnames[parnames.index('ScPoint')] = 'measurement_nr'
        if 'Sc Point' in parnames:
            parnames[parnames.index('Sc Point')] = 'measurement_nr'
        if 'OxfordSetpoint' in parnames:
            parnames[parnames.index('OxfordSetpoint')] = 'oxford_itc502_temperature1_set_K'
        if 'OxfordSetPoint' in parnames:
            parnames[parnames.index('OxfordSetPoint')] = 'oxford_itc502_temperature1_set_K'
        if 'ITC' in parnames:
            parnames[parnames.index('ITC')] = 'oxford_itc502_temperature1_set_K'
        if 'ITC_WAIT_TIME' in parnames:
            parnames[parnames.index('ITC_WAIT_TIME')] = 'oxford_itc502_wait_time_sec'
        if 'OXITCVALUE1' in parnames:
            parnames[parnames.index('OXITCVALUE1')] = 'oxford_itc502_temperature1_K'
        if 'ITCVALUE1' in parnames:
            parnames[parnames.index('ITCVALUE1')] = 'oxford_itc502_temperature1_K'
        if 'OXITCPB' in parnames:
            parnames[parnames.index('OXITCPB')] = 'oxford_itc502_P_set'
        if 'OXITCIT' in parnames:
            parnames[parnames.index('OXITCIT')] = 'oxford_itc502_I_set'
        if 'OXITCDT' in parnames:
            parnames[parnames.index('OXITCDT')] = 'oxford_itc502_D_set'
        if 'OXITCPOW' in parnames:
            parnames[parnames.index('OXITCPOW')] = 'oxford_itc502_heater_voltage_%'
        if 'oxford_temperaturetolerance_K' in parnames:
            parnames[parnames.index('oxford_temperaturetolerance_K')] = 'oxford_temperature_tolerance_K'
        if 'WAVELENGTH' in parnames:
            parnames[parnames.index('WAVELENGTH')] = 'monochromator_wavelength_AA'
        if 'Mon' in parnames:
            parnames[parnames.index('Mon')] = 'monitor_counts'
        if 'Detsum' in parnames:
            parnames[parnames.index('Detsum')] = 'detector_sum_counts'
        if 'StartT' in parnames:
            parnames[parnames.index('StartT')] = 'start_time_yyyy-mm-dd_hh:mm:ss'
        if 'MeasT' in parnames:
            parnames[parnames.index('MeasT')] = 'measuring_time_sec'
        if 'DeltaT' in parnames:
            parnames[parnames.index('DeltaT')] = 'oxford_temperature_tolerance_K'
        if 'USMCPOSA' in parnames:
            parnames[parnames.index('USMCPOSA')] = 'monochromator_angle_deg'
        if 'XangleSet' in parnames:
            parnames[parnames.index('XangleSet')] = 'monochromator_angle_set_deg'
#        vartype['DeltaT']='f4'
#        vartype['XangleSet']='f4'
        return(parnames)

    def convert_header2016(header2016):
        header={}
        if 'FILEFORMAT' in header2016:
            header['file_format_version'] = header2016['FILEFORMAT'] 
        if 'npars' in header2016:
            header['file_format_table_constants_nr_o_rows'] = header2016['npars'] 
        if 'ncolumnall' in header2016:
            header['file_format_table_variables_nr_o_columns'] = header2016['ncolumnall'] 
        if 'npoints' in header2016:
            header['file_format_table_variables_nr_o_rows'] = int(header2016['npoints'])+1 # since oct2017 the whole table, including the 'header line' is counted in nr_o_rows 
        if 'data filename' in header2016:
            header['file_name'] = header2016['data filename']
        if 'Scan filename' in header2016:
            header['self.configuration_filename'] = header2016['Scan filename']
        if 'Title' in header2016:
            header['experiment_name'] = header2016['Title']
        if 'Comment' in header2016:
            header['experiment_comment'] = header2016['Comment']
        if 'Sample' in header2016:
            header['experiment_sample_name'] = header2016['Sample']
        if 'User' in header2016:
            header['experiment_user_name'] = header2016['User']
        if 'Extension' in header2016:
            header['experiment_sample_environment'] = header2016['Extension']
        if 'NrScanPoints' in header2016:
            header['measurement_nr_o_points_set'] = header2016['NrScanPoints']
        if 'start date time' in header2016:
            header['start_time_yyyy-mm-dd_hh:mm:ss'] = header2016['start date time']
        if 'end date time' in header2016:
            header['end_time_yyyy-mm-dd_hh:mm:ss'] = header2016['end date time']
        if 'RateTime(s)' in header2016:
            header['detector_rate_sampletime_sec'] = header2016['RateTime(s)']
        if 'Min Rate' in header2016:
            header['detector_minimal_rate'] = header2016['Min Rate']
        if 'Preset Time(s)' in header2016:
            header['measurement_time_set_sec'] = header2016['Preset Time(s)']
        if 'Preset Monitor' in header2016:
            header['monitor_counts_stop'] = header2016['Preset Monitor']
        if 'Variable dac' in header2016:
            header['variable parameter'] = header2016['Variable dac']
        if 'Start Value' in header2016:
            header['start_value'] = header2016['Start Value']
        if 'End Value' in header2016:
            header['end_value'] = header2016['End Value']
        return(header)

    filename = args[0]
    file_format_version = args[1]
    measurement_type = args[2]
    a=[lines for lines in open(filename).readlines()]
    measurement_type = a[0].split('\t')[1]
    file_format_table_constants_nr_o_rows = int(a[1].split('\t')[1])
    ncolumnall = int(a[2].split('\t')[1])
    ncolumscpts = int(a[3].split('\t')[1])
    npoints = int(a[4].split('\t')[1])
    # read in header
    header2016={}
    for i in range(file_format_table_constants_nr_o_rows):
        # python-created pearl file contain a \n that you want to get rid off
        header2016[a[i].split('\t')[0]] = a[i].split('\t')[1].replace('\n','')
        # header now contains the FIRSTS file_format_table_constants_nr_o_rows number of lines of the file
        # -> the ScPoint line should be at (file_format_table_constants_nr_o_rows+1) and subsequent lines contain
        # the temperature or monochromator values.
    header = convert_header2016( header2016 )
    if 'measurement_nr_o_points_set' in header:
        measurement_nr_o_points_set = int(header['measurement_nr_o_points_set'])
#    elif 'Nrsubsetpoints' in header:
#        measurement_nr_o_points_set = int(header['Nrsubsetpoints'])
    measuring_time_sec = np.zeros((npoints,1))
    det = np.zeros((detector_nr_o_pixels, npoints))
    mon = np.zeros((detector_nr_o_pixels, npoints))
    parnames2016 = a[file_format_table_constants_nr_o_rows].split('\t') # the length changes depending on what you 'scan'
    parnames2016 = [x.strip('\n') for x in parnames2016] # remove the \n from 'MeasT'
    parnameunits = a[file_format_table_constants_nr_o_rows+1].split('\t') # the length changes depending on what you 'scan'
    parnameunits = [x.strip('\n') for x in parnameunits] # remove the \n from 's'
    parnames = convert_parnames2016( parnames2016 )
    # set up dtype for np.zeros which will be filled with the SCAN values
    mydtype = []
    for i in np.arange(ncolumscpts):
        mydtype.append((parnames[i], vartype[parnames[i]])) 
    pars = np.zeros(npoints, dtype = mydtype)
    for n in range(npoints):
        pars[n] = tuple(a[file_format_table_constants_nr_o_rows+2+n].split('\t')[0:ncolumscpts])
    measuring_time_sec = pars['measuring_time_sec']
    Nskip = file_format_table_constants_nr_o_rows + npoints + 4
    data = np.genfromtxt(filename, skip_header=Nskip)[:,1:]
    x = data[:,0]
    y_raw = data[:,1:]
    label = '(' + header['file_name'].split('.txt')[0] + ') ' + header['experiment_name'] + ' ' + header['experiment_sample_name'] + ' ' + header['experiment_comment']
    return x,y_raw,header,file_format_table_constants_nr_o_rows,ncolumscpts,npoints,measuring_time_sec,mon,label,parnames,parnameunits,pars

def read_2017file(filename, file_format_version):
    a=[lines for lines in open(filename).readlines()]
    measurement_type = a[1].split('\t')[1]
    file_format_table_constants_nr_o_rows = int(a[2].split('\t')[1])
    file_format_table_constants_nr_o_columns = int(a[3].split('\t')[1])
    file_format_table_variables_nr_o_rows = int(a[4].split('\t')[1])
    file_format_table_variables_nr_o_columns = int(a[5].split('\t')[1])
    file_format_table_detector_nr_o_rows = int(a[6].split('\t')[1])
    file_format_table_detector_nr_o_columns = int(a[7].split('\t')[1])
    measurement_nr_o_points = int(a[8].split('\t')[1])
    # read in header
    header={}
    for i in range(file_format_table_constants_nr_o_rows):
        header[a[i].split('\t')[0]] = a[i].split('\t')[1].replace('\n','')
    #
    measuring_time_sec = np.zeros((measurement_nr_o_points,1))
    det = np.zeros((file_format_table_detector_nr_o_rows, measurement_nr_o_points))
    mon = np.zeros((file_format_table_detector_nr_o_rows, measurement_nr_o_points))
    parnames = a[file_format_table_constants_nr_o_rows].split('\t') # the length of table_variables changes depending on what you 'scan'
    parnames = [x.strip('\n') for x in parnames] # remove the \n from 'measuring_time_sec'
    # set up dtype for a np.zeros which will be filled with the VARIED values (so this is from table_variables)
    mydtype = [] # the table_variables can contain time/date, counts, wavelength settings, etc. Not all 'floats'!
    for i in np.arange(len(parnames)):
        mydtype.append((parnames[i], vartype2017[parnames[i]])) 
    pars = np.zeros(measurement_nr_o_points, dtype = mydtype)
    #for n in range(measurement_nr_o_points):
    for n in range(file_format_table_variables_nr_o_rows-1): # -1 because the nr_o_rows includes the 'parnames' line
        pars[n] = tuple(a[file_format_table_constants_nr_o_rows+1+n].split('\t')[0:file_format_table_variables_nr_o_columns])
    measuring_time_sec = pars['measuring_time_sec']
    Nskip = file_format_table_constants_nr_o_rows + measurement_nr_o_points + 2 #  +2 = +1 for parnames line and +1 for 'pixel_nr ...' line
    data = np.genfromtxt(filename, skip_header=Nskip)[:,1:] # this is the actual measured data
    x = data[:,0]
    y_raw = data[:,1:]
    label = '(' + header['file_name'].split('.txt')[0] + ') ' + header['experiment_name'] + ' ' + header['measurement_sample_name'] + ' ' + header['measurement_comment'] + ' ' + header['measurement_sample_environment']
    return x, y_raw, header, file_format_table_constants_nr_o_rows, measurement_nr_o_points, measuring_time_sec, mon, label, parnames, pars

def newread_file(filename):
    """ This reads the 'new' file format """
    a=[lines for lines in open(filename).readlines()]
    file_format_table_constants_nr_o_rows = int(a[0].split('\t')[1])
    ncols = int(a[1].split('\t')[1])
    npoints = int(a[2].split('\t')[1])
    def convert_headernew(headernew):
        header={}
        if 'FILEFORMAT' in headernew:
            header['file_format_version'] = headernew['FILEFORMAT'] 
        if 'npars' in headernew:
            header['file_format_table_constants_nr_o_rows'] = headernew['npars'] 
        if 'ncolumnall' in headernew:
            header['file_format_table_variables_nr_o_columns'] = headernew['ncolumnall'] 
        if 'npoints' in headernew:
            header['file_format_table_variables_nr_o_rows'] = int(headernew['npoints'])+1 # since oct2017 the whole table, including the 'header line' is counted in nr_o_rows 
        if 'data filename' in headernew:
            header['file_name'] = headernew['data filename']
        if 'Scan filename' in headernew:
            header['self.configuration_filename'] = headernew['Scan filename']
        if 'Title' in headernew:
            header['experiment_name'] = headernew['Title']
        if 'Comment' in headernew:
            header['experiment_comment'] = headernew['Comment']
        if 'Sample' in headernew:
            header['experiment_sample_name'] = headernew['Sample']
        if 'User' in headernew:
            header['experiment_user_name'] = headernew['User']
        if 'Extension' in headernew:
            header['experiment_sample_environment'] = headernew['Extension']
        if 'NrScanPoints' in headernew:
            header['measurement_nr_o_points_set'] = headernew['NrScanPoints']
        if 'start date time' in headernew:
            header['start_time_yyyy-mm-dd_hh:mm:ss'] = headernew['start date time']
        if 'end date time' in headernew:
            header['end_time_yyyy-mm-dd_hh:mm:ss'] = headernew['end date time']
        if 'RateTime(s)' in headernew:
            header['detector_rate_sampletime_sec'] = headernew['RateTime(s)']
        if 'Min Rate' in headernew:
            header['detector_minimal_rate'] = headernew['Min Rate']
        if 'Preset Time(s)' in headernew:
            header['measurement_time_set_sec'] = headernew['Preset Time(s)']
        if 'Preset Monitor' in headernew:
            header['monitor_counts_stop'] = headernew['Preset Monitor']
        if 'Variable dac' in headernew:
            header['variable parameter'] = headernew['Variable dac']
        if 'Start Value' in headernew:
            header['start_value'] = headernew['Start Value']
        if 'End Value' in headernew:
            header['end_value'] = headernew['End Value']
        return(header)

    # read in header
    headernew={}
    for i in range(file_format_table_constants_nr_o_rows):
        headernew[a[i].split('\t')[0]] = a[i].split('\t')[1]
    header = convert_headernew( headernew)
    measuring_time_sec = np.zeros((npoints,1))
    det = np.zeros((detector_nr_o_pixels, npoints))
    mon = np.zeros((detector_nr_o_pixels, npoints))
    items=len(a[file_format_table_constants_nr_o_rows].strip().split('\t'))
    if items==6:
        pars = np.empty((npoints),dtype=[('measurement_nr','i4'), ('monochromator_wavelength_AA','f4'), ('monitor_counts','f4'), ('detector_sum_counts','f4'), ('start_time_yyyy-mm-dd_hh:mm:ss','S19'), ('measuring_time_sec','f4')])
    if items==7:
        pars = np.empty((npoints),dtype=[('measurement_nr','i4'), ('actualvalue','f4'), ('monochromator_wavelength_AA','f4'), ('monitor_counts','f4'), ('detector_sum_counts','f4'), ('start_time_yyyy-mm-dd_hh:mm:ss','S19'), ('measuring_time_sec','f4')])
    elif items==8:
        pars = np.empty((npoints),dtype=[('measurement_nr','i4'), ('setvalue','f4'), ('actualvalue','f4'), ('monochromator_wavelength_AA','f4'), ('monitor_counts','f4'), ('detector_sum_counts','f4'), ('start_time_yyyy-mm-dd_hh:mm:ss','S19'), ('measuring_time_sec','f4')])
    for n in range(npoints):
        pars[n] = tuple(a[file_format_table_constants_nr_o_rows+2+n].split('\t')[0:items])
    measuring_time_sec = pars['measuring_time_sec']
    Nskip = file_format_table_constants_nr_o_rows + npoints + 4
    data = np.genfromtxt(filename, skip_header=Nskip)[:,1:]
    x = data[:,0]
    y_raw = data[:,1:]
    label = header['experiment_name'] + ' ' + header['experiment_sample_name'] + ' ' + header['experiment_comment']
    return x,y_raw,header,file_format_table_constants_nr_o_rows,ncols,npoints,measuring_time_sec,mon,label,pars

def get_filetype(filename):
        filetype=''
        file_format_version=''
        measurement_type=''
        filetypeknown=False
        firstline= open(filename).readline()
        if len( filename.split('pearl_current') ) == 2:
            filetypeknown = True
            filetype = 'current'
        elif re.match('file_format_version',firstline):
            filetypeknown = True
            file_format_version = ''.join(firstline.split('\t')[1].strip('\n'))
        else:
            for line in open(filename,'r'):
                if re.search('FILEFORMAT',line):
                    file_format_version = 2016
                    measurement_type = ''.join(line.split('\t')[1].strip('\n'))
                if re.search('npar',line):
                    filetypeknown = True
                    filetype = 'new'
                if re.search('meastime',line):
                    filetypeknown = True
                    filetype = 'old'
        if filetypeknown == False:
            print('Do not understand file type')
            return
        return(filetype,file_format_version,measurement_type)

###########################
 
class oxfordT1(object):
    def __init__(self,filename, label = None, color = None, marker = None, plot=True, scale='K', timerelative=False, out=None):
        """ OxfordT1(filename, label = None, color = None, marker = None, plot=True, scale='K', timerelative=False)
        reads PEARL LV logfile (of Oxford ITC). """
        #
        import numpy as np
        import datetime as dt
        from matplotlib import dates

        creationtime=dt.datetime.now()
        creationtimestring = creationtime.strftime("creation time: %Y-%m-%d %H:%M")
        self.filename=filename
        if label is not None:
            self.label=label
        if color is not None:
            self.color=color
        if marker is not None:
            self.marker=marker
        if plot is not None:
            self.plot=plot
        self.scale=scale
        #
        a=np.genfromtxt(filename)
        tLV=a[:,0] # LabView time is 1 Jan 1904
        T=a[:,1]
        self.ylabel='temperature /K'
        if self.scale is 'C' or self.scale is 'Celcius' or self.scale is 'celcius':
            print('working with Celcius')
            self.ylabel='temperature / degC'
            T+=273.15
        zeroLV = dt.datetime(1904,1,1,0,0).timestamp()
        t = tLV + zeroLV + 1163  # 1163 offset POSIX and LV time somehow
        #dts = map(dt.datetime.fromtimestamp, t)
        dts = [dt.datetime.fromtimestamp(i) for i in t]
        fds = dates.date2num(dts) # converted

        # matplotlib date format object
        #hfmt = dates.DateFormatter('%A %d %H:%M')
        hfmt = dates.DateFormatter('%H:%M')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fds, T)

        ax.xaxis.set_major_locator(dates.MinuteLocator(interval=60))

        ax.xaxis.set_major_formatter(hfmt)
        ax.set_ylim(bottom = 0)
        plt.xticks(rotation='vertical')
        plt.subplots_adjust(bottom=.25)
        plt.xlabel('date', fontsize=6)
        plt.ylabel(self.ylabel, fontsize='small')
        plt.grid()
        plt.title(creationtimestring, fontsize=20)
        if out is not None:
            print('saving to:  '+out)
            plt.savefig(out, dpi=200)
        else:
            plt.show()
            plt.savefig('/home/lve/temp.png')

class tmp(object):
    def __init__(self, *args):
        import os
        Nlines = 6 # number of header lines in tmp files
        files = []
        header = {}
        if len(args) == 0:
            print('pearl.tmp(runnumber) will plot all tempory files with that runnumber in: {}'.format(tmpdir))
        else:
            runnumber = args[0]
            for f in os.listdir(tmpdir):
                if re.search('{}'.format(runnumber),f): 
                    files.append(tmpdir+f)
            files.sort()
            for f in files:
                dummy = open(f,'r').readlines()
                for l in np.arange(Nlines):
                    header[ dummy[l].split('\t')[0] ] = dummy[l].split('\t')[1]
                data = np.genfromtxt(f,skip_header=6)[:,[1,2]]
                plt.plot(data[:,0],data[:,1] / int(header['Measuring Time']),label='{}'.format(f))
                plt.ylabel('counts per second')
                plt.xlabel('2$\Theta$ /deg')
                plt.title( header['Title'] + ' | ' + header['Sample'] +  ' | ' + header['Comment'] )
                plt.grid()
                plt.legend()
        self.files = files


class data(object):
    numberOfDataRead = 0
    def __init__(self,filename, label = None, color = None, marker = None, det = None, plot=True, cps=True, corr=True, errorbar=True, summarize=False, x=None, par=None):
        import os.path
        self.cps = cps
        self.corr = corr
        self.errorbar = errorbar
        self.summarize = summarize # skip alot if you only want file info
        data.numberOfDataRead += 1
        if os.path.isfile(filename):
            self.filename = filename
        else:
            print('file "{}" does not exist'.format(filename))
            return
        self.filetype = None
        if type(self.filename) == str:
            self.filenamelabel=self.filename.split('.txt')[0]
        elif type(self.filename) == list:
            self.filenamelabel=''
            for f in self.filename:
                self.filenamelabel += f.split('.txt')[0]+' '
        self.measuring_time_sec = 0
        self.mon = 0
        self.file_format_version = ''
        self.measurement_type = ''
        # which file format is it?
        self.filetype, self.file_format_version, self.measurement_type = get_filetype(self.filename)
        if self.file_format_version == '18-10-2017':
            self.x_raw, self.y_raw, self.header, self.file_format_table_constants_nr_o_rows, self.measurement_nr_o_points, self.measuring_time_sec, self.mon, self.label, self.parnames, self.pars = read_2017file(self.filename,  self.file_format_version)
        elif self.filetype == 'new':
            if self.file_format_version == '':
                self.x_raw,self.y_raw,self.header,self.file_format_table_constants_nr_o_rows,self.ncols,self.measurement_nr_o_points,self.measuring_time_sec,self.mon,self.label,self.pars = newread_file(self.filename)
            elif self.file_format_version == 2016:
                self.x_raw,self.y_raw,self.header,self.file_format_table_constants_nr_o_rows,self.ncolumscpts,self.measurement_nr_o_points,self.measuring_time_sec,self.mon,self.label,self.parnames,self.parnameunits,self.pars = read_2016file(self.filename, self.file_format_version, self.measurement_type)
            else:
                print('new file type, but which file_format_version?')
                return
        elif self.filetype == 'current':
            self.x_raw, self.y_raw, self.currentfilename, self.label, self.title, self.sample, self.comment, self.measuring_time_sec, self.wavelength, self.measurement_nr_o_points, self.header, self.pars = currentread_file(self.filename)
        elif self.filetype == 'old':
            self.x_raw,self.y_raw,self.measuring_time_sec,self.label = oldread_file(self.filename)
            self.measurement_nr_o_points = 1
        if self.measurement_nr_o_points == 1:
            self.measuring_time_minutes = self.measuring_time_sec / 60
            self.measuring_time_hours = self.measuring_time_sec / 3600
            self.measuring_time_clock = str(dt.timedelta(seconds=int(self.measuring_time_sec)))
            if (self.filetype is not 'current') and (self.measurement_type != 'SINGLE,COMPILED') and hasattr(self,'header'):
                self.startdate= dt.datetime.strptime(self.header['start_time_yyyy-mm-dd_hh:mm:ss'],'%Y-%m-%d %H:%M:%S')
                self.enddate = dt.datetime.strptime(self.header['end_time_yyyy-mm-dd_hh:mm:ss'],'%Y-%m-%d %H:%M:%S')
                self.meandate = self.startdate + (self.enddate - self.startdate)/2

            if (self.measuring_time_sec <= 0):
                print('self.npoint==1 and self.measuring_time_sec<=0')
                self.cps = False
        else:
            self.measuring_time_minutes = np.zeros(self.measurement_nr_o_points)
            self.measuring_time_hours = np.zeros(self.measurement_nr_o_points)
            self.measuring_time_clock = []
            for i in np.arange(len(self.measuring_time_sec)):
                self.measuring_time_minutes[i] = self.measuring_time_sec[i] / 60
                self.measuring_time_hours[i] = self.measuring_time_sec[i] / 3600
                self.measuring_time_clock.append(str(dt.timedelta(seconds=int(self.measuring_time_sec[i]))))
            if self.measuring_time_sec.min()<=0:
                print('--- data(): one of self.measuring_time_sec values <=0 !!!')
                self.cps = False
        if self.summarize:
            return
        if label:
            self.title = label
            self.label = label
        if color:
            self.plotcolor = color
        #self.marker = next(mrk)
        self.marker = ''
        if marker:
            self.marker = marker
        self.graphhandle=[]
        if det == None:  # if not det efficiency from the user: use one of the below
            #self.cal=AmBe_cal # get det efficiency from this pearl module (bottom)
            #self.cal=pmma 
            #self.cal=vanadium 
            #self.cal=assembled_vanadium # use 133 vana to replace 533 Bragg peaks of V
            #self.cal=nospurions_perspexed # use perspex to replace Bragg peaks of V
            self.cal = DETEFF # see the top of this file 
        self.cal = self.cal.repeat(self.measurement_nr_o_points).reshape(detector_nr_o_pixels,self.measurement_nr_o_points)
        self.treat_data()
        if x is not None: # if you want something else than 2theta as x axis
            if x is 'Q':
                self.x = self.Q
                self.x_axis_type = 'Q'
            if x is 'd':
                self.x = self.d
                self.x_axis_type = 'd'
        else:
            self.x_axis_type = 'tth'
        if par is not None:
            self.par = par
        #elif hasattr(self,'pars') and 'oxford_itc502_temperature1_set_K' in self.pars.dtype.names:
        elif hasattr(self,'pars'):
            self.par = self.pars.dtype.names[1] # table_variables: the second column contains the first variable that is being 'scanned' (mostly one single value though)
        if (plot == True):
            self.listoflines = self.plot()

    def set_xrange(self, *args):
        """ def set_xrange(20, 140) """
        if len(args) != 2:
            print('need two arguments: tth_min, tth_max')
            return
        if args[0] == args[1]:
            print('requirement: tth_max > tth_min')
            return
        else:
            if args[0] < args[1]:
                self.xmin=args[0]
                self.xmax=args[1]
            else:
                self.xmin=args[1]
                self.xmax=args[0]

            mask = (tth>self.xmin) & (tth<self.xmax)
            self.x = self.x[mask]
            self.d = self.d[mask]
            self.Q = self.Q[mask]
            if self.measurement_nr_o_points == 1:
                self.y_raw = self.y_raw[mask]
            self.cal = self.cal[mask]
            self.xmin = self.x[0] # be accurate and make min the real min
            self.xmax = self.x[-1] # be accurate and make max the real max
            self.treat_data()
            plt.clf()
            self.plot()

    def correct_pixels(self, pixels):
        """ def correct_pixels([3,6,101]) will make these 3 pixels adopt to the mean of their respective neighbouring pixels"""
        """ it should be improved to take next-nearest neighbours in case you want to correct 2 neighbouring pixels """
        if type(pixels) is not list:
            print('need one list of pixel numbers')
            return
        pixels = np.array(pixels)
        if pixels.max() > 1407 or pixels.min() < 0:
            print('correct_pixels() requires that no pixel numbers are <0 or >1407')
            return
        else:
            pixels.sort()
            pixels = np.unique(pixels)
            # want to stick to original number of pixels: how many are there?
            orig = np.arange( len(self.x) )
            # the pixels corresponding to the correct y_raw data:
            keep = np.array( [ keep for keep in orig if keep not in pixels ] )
            ytmp = np.zeros( ( len(keep), self.y_raw.shape[1] ) )
            # loop over all data sets in this instance
            for n in np.arange( self.y_raw.shape[1] ):
                # first interpolation to get rid of 'wrong' y values:
                ytmp[:,n] = np.interp( keep, orig, self.y_raw[:,n] )
                # second interpolation to return to orig number of pixels again:
                self.y_raw[:,n] = np.interp( orig, keep, ytmp[:,n] )
            self.treat_data()

    def calc_Q_d(self):
        """ if 2theta_M setting was recognized, set d and Q axis accordingly.
        """
        doQd = False # our ground state attitude: do nothing
        if hasattr(self,'pars') and 'monochromator_wavelength_AA' in self.pars.dtype.names:
            # pre-20dec2017: wavelength=round(float(self.pars['monochromator_wavelength_AA'][0]),2)
            # in self.pars most dtypes were "f4", which is np.float32. For an == operation, both arguments need to be float32
            self.wavelength=self.pars['monochromator_wavelength_AA'].round(2)
            self.reflection=np.zeros(self.wavelength.shape)
            self.Q = np.zeros((detector_nr_o_pixels,self.wavelength.size))
            self.d = np.zeros((detector_nr_o_pixels,self.wavelength.size))
            # self.cal = np.zeros((detector_nr_o_pixels,self.wavelength.size))  # not needed because generated in class data __init__()
            doQd = True
#        elif hasattr(self,'wavelength'):
#            print('pearl.data.calc_Q_d(): already have self.wavelength')
#            self.wavelength=self.wavelength.round(2)
#            doQd = True
        if not hasattr(self,'header'):
            print('header does not exist: skip most of self.calc_Q_d()')
            doQd = False
        if doQd:
            print('pearl.data.calc_Q_d() subtracts a fitted instrumental zero for tth for 311, 533, and 733 reflections')
            self.Q133 = 4*np.pi/lambda_133_cal_AA * np.sin( (self.x - zero_133_cal_deg) / 360 * np.pi )
            self.Q533 = 4*np.pi/lambda_533_cal_AA * np.sin( (self.x - zero_533_cal_deg) / 360 * np.pi )
            self.Q733 = 4*np.pi/lambda_733_cal_AA * np.sin( (self.x - zero_733_cal_deg) / 360 * np.pi )
            self.Q755 = 4*np.pi/lambda_755_cal_AA * np.sin( self.x / 360 * np.pi )
            self.Q955 = 4*np.pi/lambda_955_cal_AA * np.sin( self.x / 360 * np.pi )
            self.d133 = 2* np.pi / self.Q133 
            self.d533 = 2* np.pi / self.Q533 
            self.d733 = 2* np.pi / self.Q733 
            self.d755 = 2* np.pi / self.Q755 
            self.d955 = 2* np.pi / self.Q955 
            for n,wavelength in enumerate(self.wavelength):
                print('n,wavelength: {}, {:1.2f}'.format(n,wavelength))
                if (wavelength == np.float32(0.00)):
                    self.wavelength[n]=lambda_533_cal_AA
                    self.reflection[n]='unknown'
                    self.Q[:,n]=self.Q533
                    self.d[:,n]=self.d533
                if (wavelength == np.float32(2.51)):
                    self.wavelength[n]=lambda_133_cal_AA
                    self.reflection[n]=133
                    self.Q[:,n]=self.Q133
                    self.d[:,n]=self.d133
                    self.cal[:,n]=DETEFF133
                    print('detector efficiency correction NOT using 133')
                if (wavelength == np.float32(1.67)):
                    self.wavelength[n]=lambda_533_cal_AA
                    self.reflection[n]=533
                    self.Q[:,n]=self.Q533
                    self.d[:,n]=self.d533
                    self.cal[:,n]=DETEFF533
                if (wavelength == np.float32(1.33)):
                    self.wavelength[n]=lambda_733_cal_AA
                    self.reflection[n]=733
                    self.Q[:,n]=self.Q733
                    self.d[:,n]=self.d733
                    self.cal[:,n]=DETEFF733
                    print('detector efficiency correction NOT using 733')
                if (wavelength == np.float32(1.10)):
                    self.wavelength[n]=lambda_755_cal_AA
                    self.reflection[n]=755
                    self.Q[:,n]=self.Q755
                    self.d[:,n]=self.d755
                    self.cal[:,n]=DETEFF755
                    print('detector efficiency correction using 755')
                if (wavelength == np.float32(0.95)):
                    self.wavelength[n]=lambda_955_cal_AA
                    self.reflection[n]=955
                    self.Q[:,n]=self.Q955
                    self.d[:,n]=self.d955
                    self.cal[:,n]=DETEFF955
                    print('detector efficiency correction using 955')
        else:
            if hasattr(self,'header'):
                self.d = self.d533 # by default
                self.Q = self.Q533 # by default
                print('note: self.pars does not exist for this file, assuming reflection 533')
            else:
                print('header does not exist: skip most of self.do_Qd()')

    def treat_data(self):
        self.tth = tth.copy() # the calibrated X axis
        self.x = self.tth.copy() # the calibrated X axis
        # 
        self.calc_Q_d() 
        self.err_raw=np.sqrt(self.y_raw)
        self.xy_raw=np.column_stack((self.x_raw,self.y_raw)).T
        self.xye_raw=np.column_stack((self.x_raw,self.y_raw,self.err_raw)).T
        #
        # determine detector pixel efficiencies
        self.deteff_err = np.sqrt(self.cal)
        self.deteff = self.cal / self.cal.mean()
        self.deteff_err = self.deteff_err / self.cal.mean()
        self.y = self.y_raw / self.deteff
        if self.y.min() <= 0:
            print()
            print('---> %s' %self.filename)
            print('---> REPLACING Y <= 0 values with Y = 1e-10')
            print('---> THIS WILL RENDER ERROR CALCULATION MEANINGLESS')
            print('---> in the .y and .err data (also _cps)')
            self.y[self.y<=0] = 1e-10
            self.err = np.zeros(self.y.shape)
        else:
            self.err = self.y_raw * np.sqrt( (self.err_raw/self.y_raw)**2 + (self.deteff_err/self.deteff)**2 )
        self.xy  = np.column_stack((self.x, self.y)).T
        self.xye = np.column_stack((self.x, self.y, self.err)).T
        if hasattr(self,'d'):
            self.dy = np.column_stack((self.d,self.y)).T
            self.dye = np.column_stack((self.d,self.y,self.err)).T
        if hasattr(self,'Q'):
            self.Qy = np.column_stack((self.Q,self.y)).T
            self.Qye = np.column_stack((self.Q,self.y,self.err)).T
        # standard make Y equal to y, unless self.cps==True
        self.Y = self.y
        self.eY = self.err
        # counts per second
        if self.cps and self.measurement_nr_o_points==1:
            self.y_cps = self.y / self.measuring_time_sec
            self.err_cps = self.err / self.measuring_time_sec
            self.y_raw_cps = self.y_raw / self.measuring_time_sec
            self.err_raw_cps = self.err_raw / self.measuring_time_sec
            self.xy_cps = np.column_stack((self.x,self.y_cps)).T
            self.xye_cps = np.column_stack((self.x,self.y_cps,self.err_cps)).T
            if hasattr(self,'d'):
                self.dy_cps = np.column_stack((self.d,self.y_cps)).T
                self.dye_cps = np.column_stack((self.d,self.y_cps,self.err_cps)).T
            if hasattr(self,'Q'):
                self.Qy_cps = np.column_stack((self.Q,self.y_cps)).T
                self.Qye_cps = np.column_stack((self.Q,self.y_cps,self.err_cps)).T
            self.Y = self.y_cps
            self.eY = self.err_cps
        elif self.cps and self.measurement_nr_o_points>1:
            self.y_cps = np.zeros( (len(self.x), self.measurement_nr_o_points) )
            self.err_cps = np.zeros( ( len(self.x), self.measurement_nr_o_points) )
            self.y_raw_cps = np.zeros( ( len(self.x), self.measurement_nr_o_points) )
            self.err_raw_cps = np.zeros( ( len(self.x), self.measurement_nr_o_points) )
            for n in np.arange(self.measurement_nr_o_points):
                self.y_cps[:,n] = self.y[:,n] / self.measuring_time_sec[n]
                self.err_cps[:,n] = self.err[:,n] / self.measuring_time_sec[n]
                self.y_raw_cps[:,n] = self.y_raw[:,n] / self.measuring_time_sec[n]
                self.err_raw_cps[:,n] = self.err_raw[:,n] / self.measuring_time_sec[n]
                self.xy_cps = np.column_stack((self.x,self.y_cps)).T
                self.xye_cps = np.column_stack((self.x,self.y_cps,self.err_cps)).T
                if hasattr(self,'d'):
                    self.dy_cps = np.column_stack((self.d,self.y_cps)).T
                    self.dye_cps = np.column_stack((self.d,self.y_cps,self.err_cps)).T
                if hasattr(self,'Q'):
                    self.Qy_cps = np.column_stack((self.Q,self.y_cps)).T
                    self.Qye_cps = np.column_stack((self.Q,self.y_cps,self.err_cps)).T
            self.Y = self.y_cps
            self.eY = self.err_cps

    def group_intensities(self, *args, **kwargs):
        if len(self.Y.shape) is not 2:
            print('pearl.data.group_intensities() can only group when it contains more than 1 diffractograms')
            print('pearl.data.group_intensities() len(self.Y.shape) = {}'.format(len(self.Y.shape)))
            return
        if 'plot' in kwargs and kwargs['plot'] is False:
            plot = False
        else:
            plot = True
        self.nr_o_groups = len(args)
        if not hasattr(self, 'group'):
            self.group={}
        if not hasattr(self, 'group_cps'):
            self.group_cps={}
        for arg_nr, arg in enumerate(args):
            if type(arg) is not (list or str):
                print('pearl.data.group_intensities(group=XX): give list(s) of integers to group, or "[all|first|last]"')
                print('pearl.data.group_intensities(group=XX): argument {} was {}'.format(arg_nr,arg))
                print('there are {} data sets in this pearl.data').format(self.measurement_nr_o_points)
                return self.plot(yspread=True)
            else:
                if type(arg) is str and arg == ('all' or 'All' or 'ALL'):
                    self.group_cps['all'] = self.y_cps.sum(axis=1) / self.measurement_nr_o_points
                    self.group['all'] = self.y.sum(axis=1)
                elif type(arg) is list and not all( isinstance(n,int) for n in arg ):
                    print('pearl.data.group_intensities() not all in list {} are integers'.format(arg))
                    return
                else:
                    self.group_cps['{}'.format(arg)] = self.y_cps[:,arg].sum(axis=1) / len(arg)
                    self.group['{}'.format(arg)] = self.y[:,arg].sum(axis=1)
        if plot and self.cps:
            self.plot(group=self.group_cps)
        if plot and not self.cps:
            self.plot(group=self.group)

    def __sub__(self,other):
        from copy import copy as classcopy
        result = classcopy(self)
        if self.measurement_nr_o_points==1: # simple single measurements
            # NONE-cps:
            result.y = self.y - other.y
            result.y_raw = self.y_raw - other.y_raw
            if (result.y<0).any():
                result.y[result.y<0] = 1e-10
                print('pearl.data.__sub__() CHANGED NEGATIVE y VALUES to 1e-10!')
            result.err = np.sqrt( (self.err)**2 + (other.err)**2 )
            # cps:
            result.y_cps = self.y_cps - other.y_cps
            if (result.y_cps<0).any():
                result.y_cps[result.y_cps<0] = 1e-10
                print('pearl.data.__sub__() CHANGED NEGATIVE y_cps VALUES to 1e-10!')
            result.err_cps = np.sqrt(  (self.err_cps)**2 + (other.err_cps)**2 )
            result.label = ' _sub_ '.join([self.label,other.label])
            print('pearl.data.__sub__() sticks to measuring time of the left-side data')
            result.measuring_time_minutes = result.measuring_time_sec / 60
            result.measuring_time_hours = result.measuring_time_sec / 3600
            result.measuring_time_clock = str(dt.timedelta(seconds=int(result.measuring_time_sec)))
            if hasattr(self,'header') and hasattr(other,'header'):
                result.header = [self.header, other.header]
            elif not hasattr(self,'header') and hasattr(other,'header'):
                result.header = [{'title', self.title}, other.header]
            elif hasattr(self,'header') and (not hasattr(other,'header')):
                result.header = [self.header, {'title', other.title}]
            elif (not hasattr(self,'header')) and (not hasattr(other,'header')):
                result.header = [{'title', self.title},{'title', other.title}]
            result.filename = [self.filename, other.filename]
            result.filenamelabel = [self.filenamelabel, other.filenamelabel]
        elif self.measurement_nr_o_points>1 and self.measurement_nr_o_points==other.npoints: # self and other contain equal amounts of data sets
            print('pearl.data.__sub__(): subtracting each of {} data sets of {} by the corresponding data set of {}!'.format(self.measurement_nr_o_points, self ,other))
            for P in np.arange(self.measurement_nr_o_points):
                # NONE-cps:
                result.y[:,P] = self.y[:,P] - other.y[:,P]
                result.y_raw[:,P] = self.y_raw[:,P] - other.y_raw[:,P]
                if (result.y[:,P]<0).any():
                    result.y[result.y<0,P] = 1e-10
                    print('pearl.data.__sub__() CHANGED NEGATIVE y VALUES to 1e-10!')
                result.err[:,P] = np.sqrt( (self.err[:,P])**2 + (other.err[:,P])**2 )
                # cps:
                result.y_cps[:,P] = self.y_cps[:,P] - other.y_cps[:,P]
                if (result.y_cps[:,P]<0).any():
                    result.y_cps[result.y_cps<0,P] = 1e-10
                    print('pearl.data.__sub__() CHANGED NEGATIVE y_cps VALUES to 1e-10!')
                result.err_cps[:,P] = np.sqrt(  (self.err_cps[:,P])**2 + (other.err_cps[:,P])**2 )
                result.label = ' _sub_ '.join([self.label,other.label])
                print('pearl.data.__sub__() sticks to measuring time of the left-side data')
                result.measuring_time_minutes[:,P] = result.measuring_time_sec[:,P] / 60
                result.measuring_time_hours[:,P] = result.measuring_time_sec[:,P] / 3600
                result.measuring_time_clock[:,P] = str(dt.timedelta(seconds=int(result.measuring_time_sec[:,P])))
                if hasattr(self,'header') and hasattr(other,'header'):
                    result.header = [self.header, other.header]
                elif not hasattr(self,'header') and hasattr(other,'header'):
                    result.header = [{'title', self.title}, other.header]
                elif hasattr(self,'header') and (not hasattr(other,'header')):
                    result.header = [self.header, {'title', other.title}]
                elif (not hasattr(self,'header')) and (not hasattr(other,'header')):
                    result.header = [{'title', self.title},{'title', other.title}]
                result.filename = [self.filename, other.filename]
                result.filenamelabel = [self.filenamelabel, other.filenamelabel]
        elif self.measurement_nr_o_points>1 and other.npoints==1:
            print('pearl.data.__sub__(): subtracting each of {} data sets of {} by the single data set of {}!'.format(self.measurement_nr_o_points, self ,other))
            for P in np.arange(self.measurement_nr_o_points):
                # NONE-cps:
                result.y[:,P] = self.y[:,P] - other.y.squeeze()
                result.y_raw[:,P] = self.y_raw[:,P] - other.y_raw.squeeze()
                if (result.y[:,P]<0).any():
                    result.y[result.y[:,P]<0,P] = 1e-10
                    print('pearl.data.__sub__() CHANGED NEGATIVE y VALUES to 1e-10!')
                result.err[:,P] = np.sqrt( (self.err[:,P])**2 + (other.err.squeeze())**2 )
                # cps:
                if self.cps and other.cps:
                    result.y_cps[:,P] = self.y_cps[:,P] - other.y_cps.squeeze()
                    if (result.y_cps[:,P]<0).any():
                        result.y_cps[result.y_cps<0,P] = 1e-10
                        print('pearl.data.__sub__() CHANGED NEGATIVE y_cps VALUES to 1e-10!')
                    result.err_cps[:,P] = np.sqrt(  (self.err_cps[:,P])**2 + (other.err_cps.squeeze())**2 )
                else:
                    result.y_cps = None
                    result.y_err_cps = None
                result.label = ' _sub_ '.join([self.label,other.label])
                print('pearl.data.__sub__() sticks to measuring time of the left-side data')
                result.measuring_time_minutes[P] = result.measuring_time_sec[P] / 60
                result.measuring_time_hours[P] = result.measuring_time_sec[P] / 3600
                result.measuring_time_clock[P] = str(dt.timedelta(seconds=int(result.measuring_time_sec[P])))
                if hasattr(self,'header') and hasattr(other,'header'):
                    result.header = [self.header, other.header]
                elif not hasattr(self,'header') and hasattr(other,'header'):
                    result.header = [{'title', self.title}, other.header]
                elif hasattr(self,'header') and (not hasattr(other,'header')):
                    result.header = [self.header, {'title', other.title}]
                elif (not hasattr(self,'header')) and (not hasattr(other,'header')):
                    result.header = [{'title', self.title},{'title', other.title}]
                result.filename = [self.filename, other.filename]
                result.filenamelabel = [self.filenamelabel, other.filenamelabel]
        if self.cps and other.cps:
            result.Y = result.y_cps
            result.eY = result.err_cps
        else:
            result.Y = result.y
            result.eY = result.err
        result.plot()
        return result

    def __add__(self,other):
        from copy import copy as classcopy
        result = classcopy(self)
        result.y = self.y + other.y
        #result.y_raw = self.y_raw + other.y_raw
        result.err = np.sqrt(result.y)
        #result.err_raw = np.sqrt(result.y_raw)
        result.measuring_time_sec = self.measuring_time_sec + other.measuring_time_sec
        result.y_cps = result.y / result.measuring_time_sec
        #result.y_cps_raw = result.y_raw / result.measuring_time_sec
        result.err_cps = result.err / result.measuring_time_sec
        #result.err_cps_raw = result.err_raw / result.measuring_time_sec
        result.label = ' _add_ '.join([self.label,other.label])
        result.measuring_time_minutes = result.measuring_time_sec / 60
        result.measuring_time_hours = result.measuring_time_sec / 3600
        result.measuring_time_clock = str(dt.timedelta(seconds=int(result.measuring_time_sec)))
        if hasattr(self,'header') and hasattr(other,'header'):
            result.header = [self.header, other.header]
        elif not hasattr(self,'header') and hasattr(other,'header'):
            result.header = [{'title', self.title}, other.header]
        elif hasattr(self,'header') and (not hasattr(other,'header')):
            result.header = [self.header, {'title', other.title}]
        elif (not hasattr(self,'header')) and (not hasattr(other,'header')):
            result.header = [{'title', self.title},{'title', other.title}]
        result.filename = [self.filename, other.filename]
        result.filenamelabel = [self.filenamelabel, other.filenamelabel]
        result.plot()
        return result

    def __mul__(self,other):
        from copy import copy as classcopy
        # new instance of self
        if type(self) == data and ( type(other) == int or type(other) == float ):
            multi = classcopy(self)
            multi.y = self.y * other
            multi.y_raw = self.y_raw * other
            multi.err = self.err * other
            multi.err_raw = self.err_raw * other
            multi.y_cps = multi.y / multi.measuring_time_sec
            multi.y_cps_raw = multi.y_raw / multi.measuring_time_sec
            multi.err_cps = multi.err / multi.measuring_time_sec
            multi.err_cps_raw = multi.err_raw / multi.measuring_time_sec
            multi.label = '{:s} *{:f}'.format(self.label,other)
            multi.measuring_time_minutes = multi.measuring_time_sec / 60
            multi.measuring_time_hours = multi.measuring_time_sec / 3600
            multi.measuring_time_clock = str(dt.timedelta(seconds=int(multi.measuring_time_sec)))
            multi.plot()
            return multi
        else:
            print('pearl.data.__mul__(self,other): self should be "data" and other should be a number')

    def __rmul__(self,other):
        from copy import copy as classcopy
        # when you arrive here, self has become 'data' and other 'int/float'
        # so now you can just call the __mul__ function.
        return self.__mul__(other)

    def __div__(self,other):
        if type(self) == data and ( type(other) == int or type(other) == float ):
            multiplier = 1.0/other
            return self.__mul__(multiplier)
        else:
            print('pearl.data.__div__(self,other) assumes self to be pearl.data and other to be a number')

    def __rdiv__(self,other):
        if type(self) == data and type(other) == data:
            print('pearl.data.__rdiv__(self,other) is a blunt division and might not calculate what you expected!')
            divided = classcopy(self)
            divided.y = self.y / other.y
            divided.y_raw = self.y_raw / other.y_raw
            divided.err = divided.y * np.sqrt( (self.err/self.y)**2  + (other.err/other.y)**2 )
            divided.err_raw = divided.y_raw * np.sqrt( (self.err_raw/self.y_raw)**2  + (other.err_raw/other.y_raw)**2 )
            divided.y_cps = self.y_cps / other.y_cps
            divided.y_cps_raw = self.y_cps_raw / other.y_cps_raw
            divided.err_cps = divided.y_cps * np.sqrt( (self.err/self.y)**2  + (other.err/other.y)**2 )
            divided.err_cps_raw = divided.y_cps_raw * np.sqrt( (self.err_raw/self.y_raw)**2  + (other.err_raw/other.y_raw)**2 )
            divided.label = '{:s} / {:s}'.format(self.label,other.label)
            divided.measuring_time_sec = self.measuring_time_sec / other.measuring_time_sec
            divided.measuring_time_minutes = divided.measuring_time_sec / 60
            divided.measuring_time_hours = divided.measuring_time_sec / 3600
            divided.measuring_time_clock = str(dt.timedelta(seconds=int(divided.measuring_time_sec)))
            divided.plot()
            return divided
        else:
            print('pearl.data.__rdiv__(self,other) assumes both self and other to be class pearl.data')

    def __radd__(self,other):
        if other == 0:
            return self
        else:
            self.__add__(other)

    def plot(self, label=None, marker=None, markersize=None, cps=None, corr=None, color=None, x=None, par=None, yspread=False, contour=False, group=None):
        """ def plot(self, label=None, marker=None, markersize=None, cps=None, corr=None, color=None, x=None, par=None):"""
        if contour is True:
            self.contour = True
        else:
            self.contour = False
        if yspread is True: # plot all measurement 'smeared out vertically' for overview
            current_fig = plt.get_fignums()[-1]
            plt.figure()
            plt.ylim([0,self.measurement_nr_o_points+2])
            CM=plt.cm.Spectral(np.linspace(0,1,self.measurement_nr_o_points))
            for n in np.arange(self.measurement_nr_o_points): 
                plt.plot(self.x, self.Y[:,n] / self.Y[:,n].max() + n, color=CM[n,:])
                plt.text(self.x[-1]*1.05, n , '{}'.format(n))
            plt.title(self.label)
            plt.figure(current_fig)
        if label:
            self.label = label
        print('pearl.data.plot(): x = ')
        print(x)
        print(self.x)
        print(self.x.shape)
        if x is None: # nothing asked within a pearl.data.plot() call
            Xhere=self.x
            if self.x_axis_type is 'tth':
                xlabel='2$\Theta$ /deg'
            elif self.x_axis_type is 'Q':
                xlabel = 'Q /$\AA^{-1}$'
            elif self.x_axis_type is 'd':
                xlabel = 'd /$\AA$'
        elif x is '2theta' or x is 'tth' or x is 'twotheta' or x is '2th':
            Xhere=self.tth
            xlabel='2$\Theta$ /deg'
        elif x is 'Q':
            print('pearl.data.plot(): Q.shape = {}'.format(self.Q.shape))
            Xhere=self.Q
            xlabel='Q /$\AA^{-1}$'
        elif x is 'd':
            Xhere=self.d
            xlabel='d-spacing /$\AA$'
        if marker:
            self.marker = marker
        if marker:
            self.marker = marker
        if markersize:
            self.markersize = markersize
        if cps:
            self.cps = cps
            self.Y = self.y_cps
        if corr:
            self.corr = corr
        if color:
            self.plotcolor = color
        if self.measurement_nr_o_points == 1:
            if self.errorbar:
                    self.graphhandle.append(plt.errorbar(Xhere, self.Y, self.eY, markersize=3, label=self.label))
                    if self.cps and self.corr:
                        plt.ylabel('counts per second')
                    else:
                        plt.ylabel('counts')
            else:
                    self.graphhandle.append(plt.plot(Xhere, self.Y, markersize=3, label=self.label))
                    if self.cps and self.corr:
                        plt.ylabel('counts per second')
                    else:
                        plt.ylabel('counts')
        elif group is None: # if npoints>1 , more than one measurement in one file and plot() not called with group=
            #CM=plt.cm.viridis(np.linspace(0,1,self.measurement_nr_o_points))
            CM=plt.cm.Spectral(np.linspace(0,1,self.measurement_nr_o_points))
            if par is None: # this par should be the one called by pearl.data.plot(par='bla'), not pearl.data(par='bla')
                parindex = self.pars.dtype.names.index(self.par)  # self.par was already set in the data() routine
            elif par in self.pars.dtype.names:     
                parindex = self.pars.dtype.names.index(par)
            else:
                print('pearl.data.plot(): {} is not in my parnames; cannot plot against this parameter'.format(par))
                parindex = 0
            for P in np.arange(self.measurement_nr_o_points):
                LABEL = '{:03n}: {:s}={:3.1f}'.format(P,self.par,self.pars[P][parindex])
                if self.errorbar:
                    self.graphhandle.append(plt.errorbar(Xhere[:,P], self.Y[:,P], self.eY[:,P], markersize=3, label=LABEL, color=CM[P,:]))
                    if self.cps and self.corr:
                        plt.ylabel('counts per second')
                    else:
                        plt.ylabel('counts')
                else:
                    self.graphhandle.append(plt.plot(Xhere[:,P], self.Y[:,P], self.eY[:,P], markersize=3, label=LABEL, color=CM[P,:]))
                    if self.cps and self.corr:
                        plt.ylabel('counts per second')
                    else:
                        plt.ylabel('counts')
            if self.contour:
                plt.title(self.label)
                current_fig = plt.get_fignums()[-1]
                plt.figure()
                T= [ t[parindex] for t in self.pars ]
                plt.contourf(self.x, T, self.Y.T)
                plt.title(self.label)
                plt.xlabel(xlabel)
                plt.ylabel('scan variable')
                plt.figure(current_fig)

        else: # group is not None, so plot each of the group element in black:
            for g in group:
                self.graphhandle.append(plt.plot(Xhere[:,g], group[g] , markersize=3, label=g, color='k', zorder=3))
        self.linehandle = self.graphhandle[0][0]
        if self.errorbar:
                self.errorbarhandle = self.graphhandle[0][1]
        plt.ylim( [ 0 , plt.gca().get_ylim()[1] ] )
        plt.legend()
        plt.grid(True)
        plt.xlabel(xlabel)
        return self.graphhandle

    def scale_to(self, other):
        """ scale_to(other): modifies your y data sets as you = p[0] + p[1]*other
        returns p,pconv,newfig """
        newfig = plt.figure()
        from scipy.optimize import curve_fit
        def fit(x,*p):
            return p[0] + p[1] * yme
        x = self.x.flatten()
        if not self.cps:
            yme = self.y.flatten()
            yother = other.y.flatten()
            lineself=self.plot(cps=False)
            lineother=other.plot(cps=False)
        else:
            yme = self.y_cps.flatten()
            yother = other.y_cps.flatten()
            lineself=self.plot(cps=True)
            lineother=other.plot(cps=True)
        p0 = [0,1]
        p, pconv = curve_fit(fit, x, yother, p0)
        self.y = fit(x,*p)
        if self.cps:
            self.y_cps = fit(x,*p)
        selfcolor=lineself[0].get_children()[0].get_color()
        plt.plot(x, fit(x,*p), '--', color=selfcolor, label='overlay: fit({})= {:.5f} + {:.3f} * {}'.format(self.header['file_name'].strip('./txt'),p[0], p[1], other.header['file_name'].strip('./txt')))
        plt.legend()
        return p,pconv,newfig


    def fit_gaussianBKG(self, tth_min=11, tth_max=158):
        """ def fit_gaussianBKG(tth_min=11, tth_max=158) """
        mask = (self.x>tth_min) & (self.x<tth_max)
        xnew = self.x[ mask ]
        ynew = self.y[ mask ]
        ynew_cps = self.y_cps[ mask ]
        ynew_cps= self.y_cps[ mask ]
        p0 = guess_pars_gaussian_bkg( xnew, ynew )
        p0_cps = guess_pars_gaussian_bkg( xnew, ynew_cps )
        p0_cps= guess_pars_gaussian_bkg( xnew, ynew_cps)
        from scipy.optimize import curve_fit
        p, pcov = curve_fit( gaussian_bkg, xnew, ynew, p0=p0 )
        p_cps, pcov_cps = curve_fit( gaussian_bkg, xnew, ynew_cps, p0=p0_cps)
        p_cps, pcov_cps= curve_fit( gaussian_bkg, xnew, ynew_cps, p0=p0_cps)
        #plt.plot(xnew, gaussian_bkg(xnew, *p0_cps),'--b', label='pars0')
        plt.plot(xnew, gaussian_bkg(xnew, *p_cps),'--k')
        print(self.label)
        print('mu0 = %3.3g amp = %3.3g sigma = %3.3g bkg = %3.3g' %(p_cps[0], p_cps[1], p_cps[2], p_cps[3]) )
        plt.show()
        #return ( (p,pcov), (p_cps,pcov_cps), (p_cps,pcov_cps) )
        return ( (p,p_cps,p_cps), (pcov,pcov_cps,pcov_cps) )

    def findpeaks(self):
        from scipy.signal import find_peaks_cwt
        width=np.array([4,5,6])
        self.peakindices = find_peaks_cwt(self.y, np.asarray(width), min_snr=0.4)
        self.peakx = self.x[self.peakindices]
        self.peaklabels = [ '%.1f' % n for n in self.x[self.peakindices] ]
        ax = plt.gca()
        ax.set_xticks( self.peakx )
        ax.set_xticklabels( self.peaklabels, rotation='vertical')
        plt.gcf().canvas.draw()
        plt.grid()

    def toggleerrorbar(self):
        if self.errorbarhandle[0].get_visible():
            self.errorbarhandle[0].set_visible(False)
            self.errorbarhandle[1].set_visible(False)
            self.errorbarhandle[2].set_visible(False)
        else:
            self.errorbarhandle[0].set_visible(True)
            self.errorbarhandle[1].set_visible(True)
            self.errorbarhandle[2].set_visible(True)
        plt.gcf().canvas.draw()


    def x2ticksBanks(self):
        nsubbanks = 22
        x2ticklabels=np.arange(nsubbanks)
        xticks = []
        for sb in range(nsubbanks):
            xticks.append(self.x[sb*64])
        xticks = np.asarray(xticks)
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        self.oldxticks = ax1.get_xticks()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(x2ticklabels,rotation=90)
        plt.grid(True)
        plt.gcf().canvas.draw()
        plt.sca(ax1)

    def x2ticksFCC(self, lattice_spacing, wavelength=lambda_533):
        """ def x2ticksFCC(self, lattice_spacing, wavelength=lambda_533) """
        x2ticklabels = []
        self.tthFCC = []
        self.hklFCC = []
        reflections = [111, 200, 220, 311, 222, 400, 331, 420, 422, 333]
        for i in reflections:
            hkl = reflection_to_hkl(i)
            lambda_over_2d = wavelength /2 / lattice_spacing * np.sqrt( hkl[0]**2 + hkl[1]**2 + hkl[2]**2 )
            if lambda_over_2d <=1:
                self.tthFCC.append( 360/np.pi * np.arcsin(lambda_over_2d) )
                self.hklFCC.append( hkl )
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        self.oldxticks = ax1.get_xticks()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(self.tthFCC)
        ax2.set_xticklabels(reflections,rotation=90)
        plt.grid(True)
        plt.gcf().canvas.draw()
        plt.sca(ax1)

    def x2ticksBCC(self, lattice_spacing, wavelength=lambda_533):
        """ def x2ticksBCC(self, lattice_spacing, wavelength=lambda_533) """
        x2ticklabels = []
        self.hklBCC = []
        self.tthBCC = []
        reflections = [110, 200, 211, 220, 310, 222, 400, 330, 420, 422, 510]
        for i in reflections:
            hkl = reflection_to_hkl(i)
            lambda_over_2d = wavelength /2 / lattice_spacing * np.sqrt( hkl[0]**2 + hkl[1]**2 + hkl[2]**2 )
            if lambda_over_2d <=1:
                self.tthBCC.append( 360/np.pi * np.arcsin(lambda_over_2d) )
                self.hklBCC.append( hkl )
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        self.oldxticks = ax1.get_xticks()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(self.tthBCC)
        ax2.set_xticklabels(reflections,rotation=90)
        plt.grid(True)
        plt.gcf().canvas.draw()
        plt.sca(ax1)

    def write_PearlFile(self,*args, **kwargs):
        """ def write_PearlFile(self,outfilename, cps=False) """
        if len(args)==1:
            outfilename=args[0]
        else:
            outfilename="%s.processedPEARLfile.txt"%self.filenamelabel
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        outfile=open(outfilename,'w',newline='\r\n')
        print('FILEFORMAT\tSINGLE,COMPILED', file=outfile)
        # need to go through these parameters in an ordered way, because otherwise the read-in routine cannot file file_format_table_constants_nr_o_rows (for instance)
        orderedParList = [ "npar", "ncolumnall", "ncolumscpts", "npoints", "data filename", "Scan filename", "Title", "Comment", "Sample", "User", "Extension", "NrScanPoints", "start date time", "end date time", "RateTime(s)", "Min rate", "Preset Time(s)", "Preset Monitor", "Variable dac", "Start Value","End Value"]
        # if this data set is simply one data file the header is a dictionary:
        if type(self.header) == dict:
            for key in orderedParList:
                print('{0}\t{1}'.format(key, self.header[key]), file=outfile)
        # if the data set is a sum/substraction of several files than the header becomes a list of dictionaries
        elif type(self.header) == list:
            idx=0 # index for selecting the numbers from the below list (which need special treatment)
            for key in orderedParList:
                if ( idx in [0,1,2,3,11]):
                    print('data set contains several headers, so taking file_format_table_constants_nr_o_rows from the first header')
                    print('{0}\t{1}'.format(key, self.header[0][key]), file=outfile)
                else:
                    string='{0}\t'.format(key)
                    #string+='_+_'.join(self.header[:][key])
                    string += '_PLUS_'.join([ p[key] for p in self.header])
                    string += '\t'
#                    for elem in np.arange(len(self.header)):
#                        string+='{0} _+_ '.format(self.header[elem][key])
#                        print('write_PearlFile; inside header loop')
#                        print(string)
#                        string.replace('\n','')
                    print(string, file=outfile)
                idx+=1
        print('measurement_nr\tmonochromator_angle_deg\tmonochromator_wavelength_AA\tmonitor_counts\tdetector_sum_counts\tstart_time_yyyy-mm-dd_hh:mm:ss\tmeasuring_time_sec', file=outfile)
        print('                        cts     cts     hh:mm:ss        s', file=outfile)
        print('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(self.pars[0][0], self.pars[0][1], self.pars[0][2], self.pars[0][3], self.pars[0][4], self.pars[0][5], self.pars[0][6]), file=outfile) 
        print('pixel_nr\tangle_tth_deg\thistogram nr 0_counts', file=outfile)
        for i in range(self.xy.T.shape[0]):
                print("{0}\t{1}\t{2}".format(i,self.xy.T[i,0],self.xy.T[i,1]), file=outfile)
        outfile.close()

    def write_GsasFile(self,*args,**kwargs):
        """ def write_GsasFile(self, outfilename, cps=False) """
        if 'par' in kwargs:
            par = kwargs['par']
        else:
            par = None
        if 'group' in kwargs:
            group = True
        else:
            group = False
        if len(args)==1:
            #outfilename="%s.raw"%args[0].split('.raw')[0].split('.dat')[0]
            outfilename=args[0]
        else:
            outfilename="%s.raw"%self.filenamelabel
        print('outfilename: {}'.format(outfilename))
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        if self.measurement_nr_o_points==1:
            outfile=open(outfilename,'w',newline='\r\n')
            print("TITLE: {0:<73s}".format(self.label[0:73]), file=outfile)
            print("{0:<80s}".format("BANK 1 1408 1408 CONS 1100 10.5 0 0 FXY"), file=outfile)
            if self.cps == True:
                outdata=np.column_stack((self.x*100,self.y_cps)) # column_stack does the transpose autom.
            else:
                outdata=np.column_stack((self.x*100,self.y_cps * self.measuring_time_sec)) # column_stack does the transpose autom.
            for i in range(outdata.shape[0]):
                data_as_string="%4.3f %8.5f"%(outdata[i,0],outdata[i,1])
                print("{0:<80s}".format(data_as_string), file=outfile)
            outfile.close()
        elif self.measurement_nr_o_points>1 and group is False:
            outfilename_base = outfilename.strip('.raw')
            if par is not None: # this par should be the one called by write_GsasFile(par='bla'), not data(par='bla')
                parindex = self.pars.dtype.names.index(par)  # self.par was already set in the data() routine
            elif self.par in self.pars.dtype.names:     
                parindex = self.pars.dtype.names.index(self.par)
            else:
                print('pearl.data.write_GsasFile(): {} is not in my parnames; cannot write files "using" this parameter'.format(par))
                parindex = 0
            for P in np.arange(self.measurement_nr_o_points):
                string_base='_point{:03d}_'.format(P)  # good to add index to this label if the scanned variable is not varying
                string = string_base + '{0:3.2f}'.format(self.pars[P][parindex])
                outfilename = outfilename_base + string + '.raw'
                outfile=open(outfilename,'w',newline='\r\n')
                print("TITLE: {0:<73s}".format(self.label[0:73]), file=outfile)
                print("{0:<80s}".format("BANK 1 1408 1408 CONS 1100 10.5 0 0 FXY"), file=outfile)
                if self.cps == True:
                    outdata=np.column_stack((self.x*100,self.y_cps)) # column_stack does the transpose autom.
                else:
                    outdata=np.column_stack((self.x*100,self.y_cps * self.measuring_time_sec)) # column_stack does the transpose autom.
                for i in range(outdata.shape[0]):
                    data_as_string="%4.3f %8.5f"%(outdata[i,0],outdata[i,1])
                    print("{0:<80s}".format(data_as_string), file=outfile)
                outfile.close()
        if group is True and hasattr(self, 'group'):
            outfilename_base = outfilename.strip('.raw')
            for n,g in enumerate(self.group):
                string = '_group{0:02d}_{1:s}'.format(n,g)  
                outfilename = outfilename_base + string + '.raw'
                outfile=open(outfilename,'w',newline='\r\n')
                print("TITLE: {0:<73s}".format(self.filename+' group: '+g), file=outfile)
                print("{0:<80s}".format("BANK 1 1408 1408 CONS 1100 10.5 0 0 FXY"), file=outfile)
                if self.cps == True:
                    outdata=np.column_stack((self.x*100,self.group_cps[g])) # column_stack does the transpose autom.
                else:
                    outdata=np.column_stack((self.x*100,self.group[g])) # column_stack does the transpose autom.
                for i in range(outdata.shape[0]):
                    data_as_string="%4.3f %8.5f"%(outdata[i,0],outdata[i,1])
                    print("{0:<80s}".format(data_as_string), file=outfile)
                outfile.close()

    def write_FullProfFile(self,*args, **kwargs):
        """ def write_FullProfFile(self, outfilename, cps=False, par='TheParameterIWantInMyFileName') """
        if 'par' in kwargs:
            par = kwargs['par']
        else:
            par = None
        if len(args)==1:
            outfilename=args[0]
        else:
            outfilename="%s.raw"%self.filenamelabel
        print('outfilename: {}'.format(outfilename))
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        if self.measurement_nr_o_points==1:
            if len(args)==1:
                outfilename=args[0]
            else:
                outfilename="%s.fp_dat"%self.filenamelabel
            if 'cps' in kwargs:
                self.cps=kwargs['cps']
            outfile=open(outfilename,'w',newline='\r\n')
            print("! TITLE: {0:s}".format(self.label), file=outfile)
            print("!", file=outfile)
            print("!", file=outfile)
            print("!", file=outfile)
            print("!", file=outfile)
            print("!", file=outfile)
            if self.cps == True:
                outdata=np.column_stack((self.x,self.y_cps, self.err_cps)) # column_stack does the transpose autom.
            else:
                outdata=np.column_stack((self.x,self.y_cps * self.measuring_time_sec, self.err)) # column_stack does the transpose autom.
            for i in range(outdata.shape[0]):
                print("{0:f}\t{1:f}\t{2:f}".format(outdata[i,0],outdata[i,1],outdata[i,2]), file=outfile)
            outfile.close()
        elif self.measurement_nr_o_points>1: # if this is a series of measurements in one file
            outfilename_base = outfilename.strip('.raw')
            if par is not None: # this par should be the one called by write_GsasFile(par='bla'), not data(par='bla')
                parindex = self.pars.dtype.names.index(par)  # self.par was already set in the data() routine
            elif self.par in self.pars.dtype.names:     
                parindex = self.pars.dtype.names.index(self.par)
            else:
                print('pearl.data.write_FullProfFile(): {} is not in my parnames; cannot write files "using" this parameter'.format(par))
                parindex = 0
            for P in np.arange(self.measurement_nr_o_points):
                string_base='_point{:03d}_'.format(P)  # good to add index to this label if the scanned variable is not varying
                string = string_base + '{0:3.2f}'.format(self.pars[P][parindex])
                outfilename = outfilename_base + string + '.raw'
                outfile=open(outfilename,'w',newline='\r\n')
                print("! TITLE: {0:s}".format(self.label), file=outfile)
                print("! par = {0:f}".format(self.pars[P][2]), file=outfile)
                print("!", file=outfile)
                print("!", file=outfile)
                print("!", file=outfile)
                print("!", file=outfile)
                if self.cps == True:
                    outdata=np.column_stack((self.x,self.y_cps[:,P], self.err_cps[:,P])) # column_stack does the transpose autom.
                else:
                    outdata=np.column_stack((self.x,self.y[:,P], self.err[:,P])) # column_stack does the transpose autom.
                for i in range(outdata.shape[0]):
                    print("{0:f}\t{1:f}\t{2:f}".format(outdata[i,0],outdata[i,1],outdata[i,2]), file=outfile)
                outfile.close()
        else:
            print('pearl.write_FullProfFile(): npoints is <=0; no data to write!')
            

    def write_XYfile(self,*args, **kwargs):
        """ def write_XYfile(self, outfilename, cps=False) """
        if len(args)==1:
            outfilename="%s.xy_dat"%args[0].split('.raw')[0].split('.dat')[0]
        else:
            outfilename="%s.xy_dat"%self.filenamelabel
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        outfile=open(outfilename,'w',newline='\r\n')
        print("# TITLE: {0:s}".format(outfilename) , file=outfile)
        outdata = self.xy_cps.T
        for i in range(outdata.shape[0]):
            print("{0:f}\t{1:f}".format(outdata[i,0],outdata[i,1]), file=outfile)
        outfile.close()

    def write_dYfile(self,*args, **kwargs):
        """ def write_dYfile(self, outfilename, cps=False) """
        if len(args)==1:
            outfilename="%s.dy_dat"%args[0].split('.raw')[0].split('.dat')[0]
        else:
            outfilename="%s.dy_dat"%self.filenamelabel
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        outfile=open(outfilename,'w',newline='\r\n')
        print("# TITLE: {0:s}".format(outfilename) , file=outfile)
        outdata = self.dy_cps.T
        for i in range(outdata.shape[0]):
            print("{0:f}\t{1:f}".format(outdata[i,0],outdata[i,1]), file=outfile)
        outfile.close()

    def write_QYfile(self,*args, **kwargs):
        """ def write_QYfile(self, outfilename, cps=False) """
        if len(args)==1:
            outfilename="%s.Qy_dat"%args[0].split('.raw')[0].split('.dat')[0]
        else:
            outfilename="%s.Qy_dat"%self.filenamelabel
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        outfile=open(outfilename,'w',newline='\r\n')
        print("# TITLE: {0:s}".format(outfilename) , file=outfile)
        outdata = self.Qy_cps.T
        for i in range(outdata.shape[0]):
            print("{0:f}\t{1:f}".format(outdata[i,0],outdata[i,1]), file=outfile)
        outfile.close()

    def write_XYEfile(self,*args, **kwargs):
        """ def write_XYEfile(self, outfilename, cps=False)"""
        if len(args)==1:
            outfilename="%s.xye_dat"%args[0].split('.raw')[0].split('.dat')[0]
        else:
            outfilename="%s.xye_dat"%self.filenamelabel
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        outfile=open(outfilename,'w',newline='\r\n')
        print("# TITLE: {0:s}".format(outfilename) , file=outfile)
        outdata = self.xye_cps.T
        for i in range(outdata.shape[0]):
            print("{0:f}\t{1:f}\t{2:f}".format(outdata[i,0],outdata[i,1],outdata[i,2]), file=outfile)
        outfile.close()

    def write_dYEfile(self,*args, **kwargs):
        """ def write_dYEfile(self, outfilename, cps=False)"""
        if len(args)==1:
            outfilename="%s.dye_dat"%args[0].split('.raw')[0].split('.dat')[0]
        else:
            outfilename="%s.dye_dat"%self.filenamelabel
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        outfile=open(outfilename,'w',newline='\r\n')
        print("# TITLE: {0:s}".format(outfilename) , file=outfile)
        outdata = self.dye_cps.T
        for i in range(outdata.shape[0]):
            print("{0:f}\t{1:f}\t{2:f}".format(outdata[i,0],outdata[i,1],outdata[i,2]), file=outfile)
        outfile.close()

    def write_QYEfile(self,*args, **kwargs):
        """ def write_QYEfile(self, outfilename, cps=False)"""
        if len(args)==1:
            outfilename="%s.Qye_dat"%args[0].split('.raw')[0].split('.dat')[0]
        else:
            outfilename="%s.Qye_dat"%self.filenamelabel
        if 'cps' in kwargs:
            self.cps=kwargs['cps']
        outfile=open(outfilename,'w',newline='\r\n')
        print("# TITLE: {0:s}".format(outfilename) , file=outfile)
        outdata = self.Qye_cps.T
        for i in range(outdata.shape[0]):
            print("{0:f}\t{1:f}\t{2:f}".format(outdata[i,0],outdata[i,1],outdata[i,2]), file=outfile)
        outfile.close()
########################### end class pearl.data()

def getdatafiles(PATH='./',contains=''):
    """
    getdatafiles(PATH='./',contains='')
    if you select via 'contains', you must give PATH too
    """
    import os
    if contains=='':
        return [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.dat']
    else:
        allfiles = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.dat']
        return [files for files in allfiles if files.count(contains)]

class disfile(object):
    def __init__(self,filename, label = None, color = None, marker = None, det = None):
        self.filename = filename
        self.filenamelabel=self.filename.split('.dat')[0]
        self.label=self.filenamelabel
        self.measuring_time_sec = 0
        self.mon = 0
        import pandas as pd
        self.data_orig = pd.read_csv(filename, sep='\t')
        self.measuring_time_sec = self.data_orig.iloc[-1,1]
        self.measuring_time_minutes = self.measuring_time_sec / 60
        self.measuring_time_hours = self.measuring_time_sec / 3600
        self.measuring_time_clock = str(dt.timedelta(seconds=self.measuring_time_sec))
        print(self.label)
        self.measuring_time_minutes = self.measuring_time_sec / 60
        self.measuring_time_hours = self.measuring_time_sec / 3600
        bankswapmask = np.flipud(np.arange(0,128))
        self.detswapmask = np.asarray([bankswapmask, bankswapmask+150, bankswapmask+300, bankswapmask+450, bankswapmask+600, bankswapmask+750, bankswapmask+900, bankswapmask+1050, bankswapmask+1200, bankswapmask+1350, bankswapmask+1500]).flatten()
        self.index = np.arange(1408)
        self.data = self.data_orig.iloc[self.detswapmask,:]

def angle_to_pixel(angle):
    for i in np.arange(np.asarray(angle).size):
        return (int( (angle-tth[0])/360*2*np.pi*1145/2.1))

def meV_to_Angstrom(meV):
    import scipy.constants as C
    E = meV *0.001 * C.e
    #E = (C.hbar * k)**2 / 2 /C.mn
    #E = (C.h / wavelength) **2 /2 /C.mn
    wavelength =  C.h / np.sqrt (2 * C.m_n * E ) 
    return (wavelength * 1e10)

def Angstrom_to_meV(Angstrom):
    import scipy.constants as C
    wavelength = Angstrom * 1e-10
    E = (C.h / wavelength) **2 /2 /C.m_n
    return (E *1000 /C.e)

tth_theoretical = np.arange(128*11)*2.1/(2*np.pi*1145)*360 + 20
dtth_theoretical = 2.1/(2*np.pi*1145)*360.0
tth0_theoretical = 11.00

tth_rad = tth /180 * np.pi


U_theory = 4 * ( A1**2*A2**2 + A1**2*beta**2 +  A2**2*beta**2 ) / ( (np.tan(mono_theta))**2 * (A1**2 + A2**2 + 4*beta**2) )
V_theory = -4 * A2**2 * ( A1**2 + 2* beta**2 ) / ( np.tan(mono_theta) * ( A1**2 + A2**2 + 4*beta**2 ) )
W_theory = ( A1**2*A2**2 + A1**2*A3**2 + A2**2*A3**2 + 4*beta**2 * ( A2**2 + A3**2 ) ) / ( A1**2 + A2**2 + 4*beta**2 )   
L_theory = A1*A2*A3*beta / ( A1**2 + A2**2 + 4*beta**2 )
U_gsas = 211.6 /100**2
V_gsas = -780 /100**2
W_gsas = 1389 /100**2
U_goubitz = 0.211835
V_goubitz = -0.371953
W_goubitz = 0.423363
U_fullprof = 0.0359110056
V_fullprof = -0.19818165498
W_fullprof = 0.31907811610
U_echidna = 0.101835
V_echidna = -0.371953
W_echidna = 0.423363
dtheta_min_deg = np.sqrt(U_theory * (np.tan(mono_theta))**2 + V_theory*np.tan(mono_theta) + W_theory)
Ahalf_theory = np.sqrt( U_theory * (np.tan(tth_rad/2))**2 + V_theory * np.tan(tth_rad/2) + W_theory )
Ahalf_gsas = np.sqrt( U_gsas * (np.tan(tth_rad/2))**2 + V_gsas * np.tan(tth_rad/2) + W_gsas )
Ahalf_goubitz = np.sqrt( U_goubitz * (np.tan(tth_rad/2))**2 + V_goubitz * np.tan(tth_rad/2) + W_goubitz )
Ahalf_echidna = np.sqrt( U_echidna * (np.tan(tth_rad/2))**2 + V_echidna * np.tan(tth_rad/2) + W_echidna )
#Ahalf_fullprof = np.sqrt( U_fullprof * (np.tan(tth_rad/2))**2 + V_fullprof * np.tan(tth_rad/2) + W_fullprof )
L = A1 * A2 * A3 / np.sqrt( A1**2 + 4* beta**2 + A2**2 ) / np.tan(mono_theta) / (lambda_533*1e10)

