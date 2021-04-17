import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import random
import scipy
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from cycler import cycler
import argparse

#argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('-l', '--location', dest = 'location', type=int, nargs=3, default=[0,0,0])
argparser.add_argument('-d', '--direction', dest = 'direction', type=int, nargs=3, default=[1,0,0])
argparser.add_argument('-e', '--energy', dest = 'energy', type = float)
argparser.add_argument('-g', '--detector_geometry', dest = 'detector_geometry', type = str)
argparser.add_argument('-o', '--output_file', dest = 'output_file', type = str)
argparser.add_argument('-b', '--debug', dest = 'debug', type = bool, default=False)
args = argparser.parse_args()

#define constants
alpha = 1/137
r_o = 2.81794e-15 #m
m_p = 938.272 #MeV/c^2
m_e = .511 #MeV/c^2
m_pi = 134.9766 #MeV/c^2
q = 1.6e-19
h = 4.136e-21 #MeV*s 
h_bar = 6.582119569e-22 #Mev*s
c  = 3e8
N_A = 6.022e23


rho = 916.7 #kg/m^3 at 0C and atmospheric pressure
Z_nuc = 7.42 #Z_eff for water
A_r = Z_nuc/.55509 #Z/A = .55509
n = 1.309 #index of refraction of ice
X_o = 1/(4*alpha*(r_o**2)*rho*(N_A/A_r)*Z_nuc*(1+Z_nuc)*np.log(183/(Z_nuc**(1/3))))
X_1 = 1433/(.916)*A_r/(Z_nuc*(Z_nuc+1)*(11.319-np.log(Z_nuc))) / 100.
X_o = X_1

abs_coeff   = 0.00853987      # according to Dima's data
abs_length  = 1 / abs_coeff
scat_coeff  = 0.0269972      # according to Dima's data
scat_length = 1 / scat_coeff

# detection_radius = .3 #half of the hole width
detection_radius = 10 # DOM oversizing
DOM_QE = .25

#energy dependence of bremsstrahlung
bs = np.loadtxt('data_bs.csv', delimiter = ',')
pp = np.loadtxt('data_pp.csv', delimiter = ',')
new_bs_sigma = []
for energy in pp[:,0]:
    i=0
    while i < len(bs[:,0])-1:
        if (bs[i][0]<= energy) & (bs[i+1][0]>= energy):
            energy_percent = np.abs((energy-bs[i+1][0])/(bs[i][0]-bs[i+1][0]))
            new_bs_sigma.append(bs[i][1] + energy_percent*(bs[i+1][1]-bs[i][1]))  
        i+=1
ratio = np.array(new_bs_sigma)/pp[:,1]
pp_prob = 1/(ratio+1)
bs_prob = 1-pp_prob

def get_prob(en):
    energies = pp[:,0]
    index = np.argmin(np.abs(energies-en))
    return pp_prob[index]

#define different particles
class Particle: 
    def __init__(self, particle_type, mass, charge, direction, location, energy, time):
        self.particle_type = particle_type
        self.mass = mass
        self.charge = charge
        self.location = location # array of [x, y, z] to describe location
        self.direction = direction # array of [theta, phi] in direction of travel
        self.theta = self.direction[0]
        self.phi = self.direction[1]
        self.energy = energy
        self.beta = np.sqrt(1-(self.mass/self.energy)**2)
        self.momentum = np.sqrt(self.energy**2-self.mass**2)/c 
        self.time = time
        
    def update_energy(self, new_energy):
        self.energy = new_energy
        self.beta = np.sqrt(1-(self.mass/self.energy)**2)
        self.momentum = np.sqrt(self.energy**2-self.mass**2)/c 

#define detector
class DOM:
    def __init__(self, location):
        self.location = location
        self.triggered = False
        self.signals = []
        self.times = []
        
    def distance_from(self, location): 
        x = location[0]
        y = location[1]
        z = location[2]
        dom_x = self.location[0]
        dom_y = self.location[1]
        dom_z = self.location[2]
        return np.sqrt((dom_x - x)**2 + (dom_y - y)**2 + (dom_z - z)**2)
    
    def trigger(self, time, num_photons=1): 
        self.triggered = True
        self.signals.append(num_photons)
        self.times.append(time)
        
    def clear_trigger(self):
        self.triggered = False
        self.signals = []
        self.times = []

class detector:
    def __init__(self, dom_list, file_name=None):
        if file_name is not None:
            saved_data = np.load(file_name, allow_pickle = True)[()]
            dom_list = []
            for load_dom in saved_data:
                dom_list.append( DOM( load_dom['location'] ) )
                dom_list[-1].triggered = load_dom['trigger']
                dom_list[-1].signals   = load_dom['signals']
                dom_list[-1].times     = load_dom['times']  
                
        self.dom_list = dom_list
        
    def num_triggered(self): 
        num_triggered = 0
        for dom in self.dom_list: #CHANGED
            if dom.triggered: 
                num_triggered += 1
        return num_triggered
    
    def volume(self):
        points = [dom.location for dom in self.dom_list]
        region = ConvexHull(points)
        return region.volume
    
    def save_detector(self, file_name):
        output_array = []
        for DOM in self.dom_list:
            output_array.append({})
            output_array[-1]['location']=DOM.location   
            output_array[-1]['trigger']=DOM.triggered
            output_array[-1]['signals']=DOM.signals
            output_array[-1]['times']=DOM.times
        np.save(file_name, output_array)
        return None

#define coordinates
#We assume that the center of IceCube is at [x,y,z] = [0,0,0]

def conv_spherical_to_cartesian(r, theta, phi):
    x = r*np.cos(phi)*np.sin(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(theta)
    return(x,y,z)

def conv_cartesian_to_spherical(x,y,z):
    r = np.sqrt(x**2+y**2+z**2) #distance to center of IceCube
    theta = np.arccos(z/np.sqrt(x**2+y**2+z**2))
    phi = np.arctan(y/x)
    return(r, theta, phi)

def add_direction(theta_i, phi_i, theta_step, phi_step):
    x_i, y_i, z_i = conv_spherical_to_cartesian(1, theta_i, phi_i)
    x_step, y_step, z_step = conv_spherical_to_cartesian(1, theta_step, phi_step)
    x_f = x_i + x_step
    y_f = y_i + y_step
    z_f = z_i + z_step
    r_f, theta_f, phi_f = conv_cartesian_to_spherical(x_f,y_f,z_f)
    return([theta_f, phi_f])

#functions for kinematics
#interact & define secondaries

def em_shower_step(particle):   
    if particle.particle_type == 'electron' or particle.particle_type =='positron':
        # move
        dx, dy, dz = conv_spherical_to_cartesian(X_o, particle.theta, particle.phi)
        particle.location = [particle.location[0]+dx,particle.location[1]+dy,particle.location[2]+dz]
        dtime = X_o / (particle.beta * c)

        #interact & define secondaries
        phi_i = random.uniform(0,np.pi)
        pair_prod = get_prob(particle.energy); 
        interaction = random.random()
        if interaction <= pair_prod: #assuming it hits a positron/electron at rest
            #print('pp')
            photon_energy = (particle.energy+particle.mass)/2
            direction1_step = [np.arccos(np.sqrt((
                particle.energy**2-particle.mass**2)/2*(photon_energy/c)**2)), phi_i]
            direction2_step = [-np.arccos(np.sqrt((
                particle.energy**2-particle.mass**2)/2*(photon_energy/c)**2)), np.pi-phi_i]
            direction1 = add_direction(particle.theta, particle.phi, 
                                       direction1_step[0], direction1_step[1])
            direction2 = add_direction(particle.theta, particle.phi, 
                                       direction2_step[0], direction2_step[1])
            photon1 = Particle('photon', 0, 0, direction1, particle.location, 
                               photon_energy, particle.time+dtime) 
            photon2 = Particle('photon', 0, 0 , direction2, particle.location, 
                               photon_energy, particle.time+dtime) 
            return(photon1,photon2)  
        else: #bremsstrahlung
            #print('bremss')
            photon_momentum = particle.energy/(2*c) #get rid of c?
            secondary_electron_momentum = np.sqrt((particle.energy/2)**2 - m_e**2)/c #get rid of c?
            electron_direction_step = [np.arccos((particle.momentum**2-photon_momentum**2)/(2*particle.momentum*secondary_electron_momentum)), phi_i]
            electron_direction = add_direction(particle.theta, particle.phi, electron_direction_step[0], electron_direction_step[1])
            photon_direction = [np.arcsin(np.sin(electron_direction_step[0])*secondary_electron_momentum/photon_momentum),np.pi - phi_i]
            electron_energy = particle.energy/2
            photon = Particle('photon', 0, 0, photon_direction, particle.location, 
                              (particle.energy)/2, particle.time+dtime)  
            secondary_electron = Particle(particle.particle_type,m_e,
                                          particle.charge,electron_direction, particle.location, 
                                          electron_energy, particle.time+dtime) 
            return(secondary_electron, photon)
    if particle.particle_type=='photon':
        #move
        dx, dy, dz = conv_spherical_to_cartesian(scat_length, particle.theta, particle.phi)
        particle.location = [particle.location[0]+dx,particle.location[1]+dy,particle.location[2]+dz]
        dtime = scat_length / (particle.beta * c)

        #interact & define secondaries for pair production
        electron_energy = particle.energy/2
        direction1 = particle.direction 
        direction2 = particle.direction 
        electron = Particle('electron', .511, -1, direction1, particle.location, 
                            electron_energy, particle.time+dtime) 
        positron = Particle('positron', .511, 1 , direction2, particle.location, 
                            electron_energy, particle.time+dtime) 
        return(electron,positron)

#functions for Cherenkov radiation
## Define RGBA to HEX
def rgba_to_hex(rgba):
    r = int(rgba[0]*255.0)
    g = int(rgba[1]*255.0)
    b = int(rgba[2]*255.0)
    return '#{:02X}{:02X}{:02X}'.format(r,g,b)

# Get color map
rainbow_cm = plt.get_cmap('nipy_spectral')
def color_map(n):
    # n = number of colors in the rainbow spectrum to return
    clist = [rainbow_cm(1.0*i/n) for i in range(n)]
    return [rgba_to_hex(ci) for ci in clist]

class VisibilityRegion:
    """
    Class representing the region in which a photon can be observed by a DOM along its path
    
    Parameters
    ----------
    step_positions: list
        List of dictionaries of structure
        {'location': float, 'direction': {'theta': float,  'phi': float}}
        where the location is the photon location and theta/phi describe the
        direction in which the photon is traveling
    viewable_radius: float
        distance from the photon location that the photon can be viewed from
        recommendation: radius of a DOM
    
    Attributes
    ----------
    regions: list
        List of Scipy ConvexHulls representing the photon's path, 
        incorporating the viewable_radius. 
    
    """
    
    def __init__(self, step_positions, viewable_radius):
        # Step positions is a list of dictionaries with keys: location, direction (theta/phi?)
        self.step_positions    = step_positions
        self.viewable_radius   = viewable_radius
        
        self.regions = []
        for i, step in enumerate(self.step_positions):
            if i == len(self.step_positions)-1: 
                continue
            else: 
                self.regions.append(ConvexHull(np.vstack((
                    self.create_circle(step['location'], 
                                       step['direction']['theta'], 
                                       step['direction']['phi']) ,
                    self.create_circle(self.step_positions[i+1]['location'], 
                                       self.step_positions[i+1]['direction']['theta'], 
                                       self.step_positions[i+1]['direction']['phi'] )
                ))))
                
    def create_circle(self, vtx, theta, phi):
        """
        Circle centered on the photon pointing in its direction of travel
        and oriented in the photon travel direction
        
        Parameters
        ----------
        vtx: float
            The 3D location position of the center of the circle
        theta: float
            The angle in zenith by which the circle is tilted. 0 = straight up
        phi: float
            The angle in azimuth by which the circle is rotated. 
        
        Returns
        ----------
        ndarray
            A numpy array of 3D coordinates for a circle of radius viewable_radius 
            with its center at vtx oriented by theta in zenith and phi azimuthally
        """
        
        # Generic circle in x-y plane (at z = 0)
        circle_thetas = np.linspace(0, 2*np.pi, 50)
        circle = np.array([
            self.viewable_radius * np.cos(circle_thetas),
            self.viewable_radius * np.sin(circle_thetas),
            [0]*len(circle_thetas),
        ]).T
        x = circle[:,0]; y = circle[:,1]; z = circle[:,2]
        
        # Manipulating and returning generic circle
        cos_t = np.cos(np.radians(theta)); sin_t = np.sin(np.radians(theta))
        cos_p = np.cos(np.radians(phi));   sin_p = np.sin(np.radians(phi))
        return np.array([
                    # R_z * R_x
#                     (x*cos_p - y*sin_p*cos_t + z*sin_p*sin_t) + vtx[0],
#                     (x*sin_p + y*cos_p*cos_t - z*cos_p*sin_t) + vtx[1],
#                                           (y*sin_t + z*cos_t) + vtx[2] 
                    # R_x * R_z
                                          (x*cos_p + y*sin_p) + vtx[0],
                    (x*cos_t*sin_p + y*cos_t*cos_p - z*sin_t) + vtx[1],
                    (x*sin_t*sin_p + y*sin_t*cos_p + z*cos_t) + vtx[2] 
                ]).T
                
        
    def contains(self, point): 
        """
        Determines if the provided point is within the photon viewability region
        
        Parameters
        ----------
        point: array
            array of format [x, y, z] defining a 3D point
        
        Returns
        ----------
        boolean
            True if the point is along the photon's path and within the photon's
            viewability region. 
            False otherwise
        """
        
        for region in self.regions:
            # From https://stackoverflow.com/questions/29311682/finding-if-point-is-in-3d-poly-in-python
            test_region = ConvexHull(
                np.concatenate((region.points, [point]))
            )
            if np.array_equal(test_region.vertices, region.vertices): 
                return True
        return False
        
    def show(self, x_axis='x', y_axis='y'): 
        """
        Prints to screen the photon path according to the specified axes
        
        Parameters
        ----------
        x_axis: str
            Photon travel axis to plot on the plot's X-Axis. 'x' if plotting 
            photon x-path on plot's x-axis. Options: 'x', 'y', 'z'
        y_axis: str
            Photon travel axis to plot on the plot's Y-Axis. 'z' if plotting 
            photon z-path on plot's y-axis. Options: 'x', 'y', 'z'
        
        Returns
        ----------
        None
        """
        
        axis_index = {'x':0, 'y':1, 'z':2}
#         ax.set_prop_cycle(cycler('color', hexclist))
        colors = color_map(len(self.regions))
        x_min = None; x_max = None
        y_min = None; y_max = None
        for i, region in enumerate(self.regions):
            x_vals = region.points[region.vertices, axis_index[x_axis]]
            y_vals = region.points[region.vertices, axis_index[y_axis]]
            plt.scatter(x_vals, y_vals, color=colors[i])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
        return None

def random_walk(photon, n_steps, detector, plot_step=False):
    
    photon_steps = [{'location': photon.location, 
                     'direction': {'theta': photon.theta,  'phi': photon.phi}}]
    for step in range(n_steps):
        absorption = random.random()
        if absorption <= scat_length / abs_length: # reverse of what Justin said
            break
       
        # Add a random angle in phi and theta to current direction
        move_theta = math.pi * random.random()   
        move_phi = 2*math.pi * random.random()
        new_theta = add_direction(photon_steps[-1]['direction']['theta'],
                                  photon_steps[-1]['direction']['phi'],
                                  move_theta, move_phi)[0]
        new_phi = add_direction(photon_steps[-1]['direction']['theta'],
                                photon_steps[-1]['direction']['phi'],
                                move_theta, move_phi)[1]
        
        # take a step in x,y,z based on new direction of travel
        dx, dy, dz = conv_spherical_to_cartesian(scat_length, new_theta, new_phi)
        new_loc = [photon_steps[-1]['location'][0]+dx,
                   photon_steps[-1]['location'][1]+dy,
                   photon_steps[-1]['location'][2]+dz]
        photon_steps.append({'location': new_loc, 
                             'direction': {'theta': new_theta,  'phi': new_phi}})
        photon.direction = [new_theta, new_phi] 
        photon.theta = new_theta
        photon.phi = new_phi
        photon.location = new_loc
        
        #Determine if DOM is within visibility region / if a DOM triggers
        visibility_region = VisibilityRegion(photon_steps[-2:], detection_radius)
        
        # Determine if DOMs trigger
        for dom in detector.dom_list:   
            if visibility_region.contains(dom.location): 
                if random.random() <= DOM_QE: 
                    dom.trigger(scat_length*step / c + photon.time)
                break
            else: 
                continue
        break
    
    if plot_step: 
        print(photon_steps[0]['location'][2], photon_steps[-1]['location'][2])
        visibility_region = VisibilityRegion(photon_steps, detection_radius)
        visibility_region.show()


    return detector

def cherenkov(charged_particle,n_steps,random_walk_n_steps, detector, debug=False): 
    #make cone & see if it intersects w/ detector, 
    # return every Cherenkov photon & its path
    angle = np.arccos(1/(n*charged_particle.beta))
    dNdx = (2*math.pi*(charged_particle.charge**2)
            *alpha*(np.sin(angle)**2)*((1/(400e-9))-(1/(700e-9))))
    N = dNdx * X_o #assuming the electron travels 1 radiation length as it emits Cherenkov radiation
    
    if debug: 
        N = 100
    
    cherenkov_photons = []
    for i in range(n_steps):
        
        if debug: print(f"\t\tSimulating Cherenkov step {i} for {int(N/n_steps)} photons")
            
        particle_dx = X_o * i / n_steps
        interaction_time = particle_dx / (charged_particle.beta * c) + charged_particle.time
        
        for photon in range(int(N/n_steps)):
            
            if debug and photon % 10 == 0: print(f"\t\t\tSimulating photon {photon}")
            
            add_angle = add_direction(charged_particle.direction[0],
                                      charged_particle.direction[1], 
                                      angle, 2*np.pi/N * i)
            
            new_photon = Particle('photon', 0, 0, add_angle, 
                                  charged_particle.location, 
                                  h*c/(450e-9), charged_particle.time + interaction_time) #assumed blue light
            cherenkov_photons.append(new_photon)
            
            if debug:
                if photon%10 == 0: 
                    detector = random_walk(new_photon, n_steps, detector, 
                                           plot_step=False) # If plot_step=True it outputs photon path to terminal
                else: 
                    detector = random_walk(new_photon, n_steps, detector)
            else:
                detector = random_walk(new_photon, n_steps, detector)

            energy_lost = N*h*c/(450e-9)
            charged_particle.update_energy(charged_particle.energy-energy_lost)
        
        # update particle location
        dx, dy, dz = conv_spherical_to_cartesian(particle_dx, 
                                                 charged_particle.theta, 
                                                 charged_particle.phi)
        charged_particle.location = [charged_particle.location[0]+dx,
                                     charged_particle.location[1]+dy,
                                     charged_particle.location[2]+dz]
        
    return(N, detector)

#Simulate Shower
def em_shower(neutrino, steps, n_cher_steps, detector, plot_shower=False, debug=False):
    time = 0.0
    electron = Particle('electron',m_e,-1,
                        neutrino.direction, neutrino.location, neutrino.energy, time)
    particle_plotters = [[np.array([neutrino.location, electron.location])]]
    incoming_particles =[electron]
    for n in range(steps):
        if debug: print(f"Simulating EM Shower step {n}")
        outgoing_particles = []
        particle_plotters.append([])
        for particle in incoming_particles:
            new_particles = em_shower_step(particle)
            for new_particle in new_particles: 
                particle_plotters[-1].append(np.array([particle.location, 
                                                       new_particle.location]))
            outgoing_particles.append(new_particles[0])
            outgoing_particles.append(new_particles[1])
        for particle in outgoing_particles:
            if particle.charge != 0:
                if debug: print(f"\tSimulating Particle {particle.particle_type}")
                N_photons, detector = cherenkov(particle, n_cher_steps, 
                                                2, detector, debug=debug)
        incoming_particles = outgoing_particles
        
    if plot_shower: 
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        colors = color_map(len(particle_plotters))
        
        for i, step in enumerate(particle_plotters): 
            for line in step: 
                ax.plot(line[:,0], line[:,1], line[:,2], color=colors[i])
        
        x_min = None; x_max = None
        y_min = None; y_max = None
        for i, region in enumerate(self.regions):
            x_vals = region.points[region.vertices, axis_index[x_axis]]
            y_vals = region.points[region.vertices, axis_index[y_axis]]
            plt.scatter(x_vals, y_vals, color=colors[i])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
        return None
        
    return detector

if args.detector_geometry == 'IceCube_small':
    X, Y, Z = np.mgrid[-20:20:2j, -20:20:2j,-20:20:2j]
elif args.detector_geometry == 'IceCube':
    X, Y, Z = np.mgrid[-500:500:21j, -500:500:21j,-500:500:21j]
locations = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
dom_list = [DOM(loc) for loc in locations]
IceCube = detector(dom_list)


#Actual Event
neutrino = Particle('electron neutrino', 0, 0, args.direction, args.location, args.energy, 0)
IceCube = em_shower(neutrino, 6, 20, IceCube, debug=args.debug)
IceCube.save_detector(args.output_file)
