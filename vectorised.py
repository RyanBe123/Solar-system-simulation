# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:20:16 2019

@author: ryanb
"""



import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


#constants
G = 6.67*10**(-11)

force_list =np.zeros((1,2))
net_force= np.zeros((1,2))
    
""" Generate a uniform array for 9 objects between 0 and 1. Multiply it by 2pi
to produce random angles for each planet in the solar system.
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
    #names = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']
    full_mass = 10**(24) * np.array([1.989*10**6,0.33,4.867,5.972,0.65,19000,570,87,100])
    #full_mass = [x * 10**(24) for x in mass]
    
    initial_distance=[0,57.9,108.2,149.6,227.9,778.3,1427,2871,4497.1]
    full_initial_distance =10**(9) * np.array(initial_distance)
    planet_positions =  random_initial_positions(full_initial_distance,len(initial_distance))[0]
    planet_angles = random_initial_positions(full_initial_distance,len(initial_distance))[1]
    
    full_initial_velocity =10**3 * np.array([0,47.4,35,29.8,24.1,13.1,9.7,6.8,5.4])
   # full_initial_velocity = [x*10**3 for x in initial_velocity]
    planet_velocities = np.zeros((9,2))
 
    planet_velocities[:,0] = np.dot(full_initial_velocity,np.cos(planet_angles))
    planet_velocities[:,1] = np.dot(full_initial_velocity,np.sin(planet_angles))

    
    
    
    planet_variables = {'mass': full_mass, 
                        'radius':full_initial_distance, 
                        'x position': planet_positions[:,0],
                        'y position': planet_positions[:,1],  
                        'initial x velocities' :planet_velocities[:,0],
                        'initial y velocities': planet_velocities[:,1]}
    planet_variables=pd.DataFrame(data = planet_variables)
   
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
    radius_distribution=np.array(radius_distribution)
    initial_angle=np.array(initial_angle)
    velocity_distribution = np.zeros((no_of_asteroids,2))
    velocity_distribution[:,0] = np.dot(np.sqrt(np.divide([G*1.989*10**(30)],radius_distribution)),np.cos(initial_angle))
    velocity_distribution[:,1] = np.dot(np.sqrt(np.divide([G*1.989*10**(30)],radius_distribution)),np.sin(initial_angle))
    asteroid_variables = {'mass' : mass_distribution,
                          'radius':radius_distribution,
                          'x position': initial_pos[:,0],
                          'y position': initial_pos[:,1],
                          'initial x velocity': velocity_distribution[:,0],
                          'initial y velocity': velocity_distribution[:,1]}
    asteroid_variables = pd.DataFrame(data = asteroid_variables)
        
    return asteroid_variables



""" call the "generate random position" function to get the initial positions
of each planet. Loop through each planet, find the distance between the planet
in question and the other bodies then determine the resulting force. Sum all the
forces in x and y to find the net force for each body in the x and y direction
"""
def grav(dataset):
    
    mass = dataset[:,0]
    mass_horizontal  = np.array([[mass],]*len(dataset)).T
    
    
    x_positions = dataset[:,2]
    y_positions=dataset[:,3]
    x_horizontal = np.array([[x_positions],]*len(dataset)).T
    y_horizontal = np.array([[y_positions],]*len(dataset)).T
    x_vertical = np.array([[x_positions],]*len(dataset))
    y_vertical = np.array([[y_positions],]*len(dataset))
    xi_minus_xj = x_horizontal-x_vertical
    yi_minus_yj = y_horizontal-y_vertical
    
    ax = -(G * mass_horizontal * (xi_minus_xj))/((xi_minus_xj**2 + yi_minus_yj**2)**(3/2)+0.0001)
    ay = -(G * mass_horizontal * (yi_minus_yj))/((xi_minus_xj**2 + yi_minus_yj**2)**(3/2)+0.0001)
    
    net_acceleration_x = np.sum(ax,axis=0)
    net_acceleration_y = np.sum(ay,axis=0)
    

    return net_acceleration_x , net_acceleration_y


    

"""Putting the resultant force into a SUVAT equation to get the direction of 
velocity for each time step. Then times the velocity by timestep to find new position
"""


def suvat(current_position,acceleration,t,u):
    #v = np.ones((len(current_position),2))
    
    s =  np.array(current_position)
    u = np.array(u)
    v = np.zeros_like(u)
    new_position = np.zeros_like(s)
    a = acceleration
    new_position = s + u*t + 0.5*a*(t**2)
    v = u + a*t
    
    return new_position, v

   

"""Testing the gravitational function for some simple bodies by  producing the 
same calculation by hand. The y axis is the same for each body therefore will
produce an infinity in the y direction.  The Suvat function is also tested here.
"""
def grav_suvat_test():
    test_position= np.zeros((3,2))
    test_position[:,0]=[-1,-2,-3]
    test_position[:,1]=[-1,-2,-3]
    test_initial_velocity =np.zeros((3,2))
    test_initial_velocity[:,0] = [0,0,0]
    test_initial_velocity[:,1] = [0,0,0]
    test_distance= [-1,-2,-3]
    test_mass = [10**(9),2*10**(9),3*10**(9)]
    test_data = {'mass':test_mass,
                 'radius':[1,1,1],
                 'x position':test_distance,
                 'y position':test_distance}
    test_data = pd.DataFrame(data = test_data)
    test_data = np.array(test_data)
    print(test_data)
    test_accel_array = np.zeros((3,2))

            
    
    net_acceleration_x=grav(test_data)[0]
    net_acceleration_y=grav(test_data)[1]
    test_accel_array[:,0] = net_acceleration_x
    test_accel_array[:,1] = net_acceleration_y
    
    test_position = suvat(test_position,test_accel_array,1,test_initial_velocity)[0]
    test_initial_velocity = suvat(test_position,test_accel_array,1,test_initial_velocity)[1]
    print(test_position)
    print(test_initial_velocity)
    return test_position,test_initial_velocity



"""Looping through each time step. Within this loop the acceleration due to grav
is found for every body by calling the grav function. The new position and velcitty
is produced for every body by calling the SUVAT equation.
"""

time_step = 60
times = []
bodies=[]
time_steps=[]
for asteroid_frequency in np.arange(3,4):
    asteroid_frequency = 4 ** (asteroid_frequency)
    bodies.append(asteroid_frequency+9)
    """Calling the planet and asteroid generation functions and joining them
    to produce an array containg all of their respetive variables. Find the
    x and y distances between each body by using vector subtraction.
    """

    planet_variables = generate_planets()
    asteroid_variables = generate_asteroids(asteroid_frequency)
    acceleration_array = np.zeros(((9+asteroid_frequency),2))
    time_start = time.time()

    planet_variables=np.array(planet_variables)
    asteroid_variables = np.array(asteroid_variables) 
    full_array=np.concatenate((planet_variables.T,asteroid_variables.T),axis=1) 
    full_array = full_array.T
    for t in np.arange(60,3.154 * 10 **(5),time_step):   
        
        acceleration_array[:,0]=grav(full_array)[0]
        acceleration_array[:,1]=grav(full_array)[1]

        
       


        new_positions = suvat(full_array[:,[2,3]],
                              acceleration_array,time_step,
                              full_array[:,[4,5]])[0]
        
        
        
        new_velocity =suvat(full_array[:,[2,3]],
                              acceleration_array,time_step,
                              full_array[:,[4,5]])[1]
        
        full_array[:,[2,3]] = new_positions
        full_array[:,[4,5]] = new_velocity
        
        

    time_end = time.time()
        
    time_taken = time_end - time_start
    times.append(time_taken)
    


print('overall',times)  
print('bodies', bodies)



