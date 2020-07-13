
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:50:51 2019
7
@author: ryanb
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from libc.math cimport sqrt



force_list =np.zeros((1,2))
net_force= np.zeros((1,2))
random_initial_positions_times =[]
generate_planets_times=[]
generate_asteroid_times=[]
grav_times=[]
suvat_times=[]


#constant
cdef:
    float G
    double [:,:] full_array
G = 6.67*10**(-11)    


""" Generate a uniform array for between 0 and 1. Multiply it by 2pi
to produce random angles for each planet in the solar system. Then find the x and y
positions of each body by using the radius, angle and simple trig
"""
def random_initial_positions(radius,frequency):
    np.random.seed(0)
    positions = np.zeros((frequency,2))
    rad = np.random.uniform(0,1,frequency)
    rad = list(rad*2*np.pi)
    for a,b in zip(rad,radius):
        x,y = b*np.cos(a),b*np.sin(a)
        positions[rad.index(a),0] = x
        positions[rad.index(a),1] = y
    return positions, rad


"""Taken data from NASA website to find the masses, velocity and radius of orbit
of the planets and sun. Called the random initial position function to generate
positions then put all the data into a dataframe
"""
def generate_planets():
   # time_start=time.time()
    #names = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    mass = [1.989*10**6,0.33,4.867,5.972,0.65,19000,570,87,100]
    full_mass = [x * 10**(24) for x in mass]
       
    initial_distance=[0,57.9,108.2,149.6,227.9,778.3,1427,2871,4497.1]
    full_initial_distance =[x * 10**(9) for x in initial_distance]
    planet_positions =  random_initial_positions(full_initial_distance,len(initial_distance))[0]
    planet_angles = random_initial_positions(full_initial_distance,len(initial_distance))[1]
    
    initial_velocity = [0,47.4,35,29.8,24.1,13.1,9.7,6.8,5.4]
    full_initial_velocity = [x*10**3 for x in initial_velocity]
    planet_velocities = np.zeros((9,2))
    for pv in np.arange(len(planet_velocities)):
        planet_velocities[pv,0] = full_initial_velocity[pv]*np.cos(planet_angles[pv])
        planet_velocities[pv,1] = full_initial_velocity[pv]*np.sin(planet_angles[pv])
    
    
    
    planet_variables = {'mass': full_mass, 
                        'radius':full_initial_distance, 
                        'x position': planet_positions[:,0],
                        'y position': planet_positions[:,1],  
                        'initial x velocities' :planet_velocities[:,0],
                        'initial y velocities': planet_velocities[:,1]}
    planet_variables=pd.DataFrame(data = planet_variables)
    
  #  time_end = time.time()
        
  #  time_taken = time_start - time_end
  #  generate_planets_times.append(time_taken)
    return planet_variables



"""Similar as previous function but use random distributions to find each variable
"""
def generate_asteroids(no_of_asteroids):  
    np.random.seed(0)
    mass_distribution = np.random.normal(loc = 10**(12), scale= 10**(5), size = no_of_asteroids)
    radius_distribution = np.random.normal(loc = 3.74*10**(11), scale =10**(8), size = no_of_asteroids)
    #velocity_distribution = np.random.normal(loc = 5.81*10*(3), scale = 2*10^(3), size = no_of_asteroids)
    initial_pos = random_initial_positions(radius_distribution,no_of_asteroids)[0]
    initial_angle = random_initial_positions(radius_distribution,no_of_asteroids)[1]
    velocity_distribution = np.zeros((no_of_asteroids,2))
    for v in np.arange(len(velocity_distribution)):
        velocity_distribution[v,0] = np.sqrt((G*1.989*10**(30))/(radius_distribution[v])) * np.cos(initial_angle[v])
        velocity_distribution[v,1] = np.sqrt((G*1.989*10**30)/(radius_distribution[v])) * np.sin(initial_angle[v])
    asteroid_variables = {'mass' : mass_distribution,
                          'radius' : radius_distribution,
                          'x position': initial_pos[:,0],
                          'y position': initial_pos[:,1],
                          'initial x velocity': velocity_distribution[:,0],
                          'initial y velocity': velocity_distribution[:,1]}
    asteroid_variables = pd.DataFrame(data = asteroid_variables)
   #     time_end = time.time()
        
#    time_taken = time_start - time_end
#    generate_asteroid_times.append(time_taken)
    return asteroid_variables



    
"""Looping through each time step. Within this loop the acceleration due to grav
is found for every body by calling the grav function. The new position and velcitty
is produced for every body by calling the SUVAT equation.
"""
def main(time_power):    
    time_step = 60
    times = []
    bodies=[]

    cdef:
        #double [:] mass
        #double [:,:] positions
        int i
        int l
        int length
       # int mj_index
        double ax
        double ay
        double mj
        double xi_minus_xj
        double yi_minus_yj
        double fx
        double fy
     #   double [:,:] force_list
        double denominator
        double [:,:] net_acceleration
        
    
    #G = 6.67*10**(-11)    
    
    
    for asteroid_frequency in np.arange(0,5):
        asteroid_frequency = 4 ** (asteroid_frequency)
        bodies.append(asteroid_frequency+9)
        """Calling the planet and asteroid generation functions to produce a dataframe
        containg their respetive variables
        """
        planet_variables = generate_planets()
        asteroid_variables = generate_asteroids(asteroid_frequency)
        acceleration_array = np.zeros(((9+asteroid_frequency),2))
        
        time_start =time.time()
        
        planet_variables=np.array(planet_variables)
        asteroid_variables = np.array(asteroid_variables) 
        full_array=np.concatenate((planet_variables.T,asteroid_variables.T),axis=1) 
        full_array = full_array.T     
        
        
        for t in np.arange(60,3.154 * 10 **(5),time_step):
            for i in np.arange(0,len(full_array)):
                length = len(full_array)
             #   mass = full_array[:,0]
              #  positions =full_array[:,[2,3]]
                
          
                for l in np.arange(0,length):
                    mj = full_array[l,0]
                    xi_minus_xj = full_array[i,2]-full_array[l,2] 
                    yi_minus_yj = full_array[i,3]-full_array[l,3]
                
                    denominator = sqrt(xi_minus_xj*xi_minus_xj + yi_minus_yj*yi_minus_yj + 0.000001)
                    ax += -G*(mj)*(xi_minus_xj)/(denominator * denominator * denominator)
                    ay +=-G*(mj)*(yi_minus_yj)/(denominator * denominator * denominator)
                                
                        
        time_end = time.time()
        time_taken = time_end-time_start
        times.append(time_taken)
    
    print('bodies',bodies)
    print('times',times)
   
    return 0
