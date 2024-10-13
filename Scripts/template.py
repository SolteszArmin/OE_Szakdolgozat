'''Ez egy template fálj, bármilyen Carla script alpaja kb. ez.'''
#Carla is very sensitive to the Python version on Windows. The recommended version is 3.7.
#################################################################################################################
#Carla Eviroment setup
import glob
import os
import sys

import carla
import pygame

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
###################################################################################################################


host_ip='localhost'
base_port=2000

actor_list = []

try:
    client= carla.Client(host_ip,base_port)
    client.set_timeout(10.0)

    world=client.get_world()
    blueprint_library=world.get_blueprint_library()


finally:
    for actor in actor_list:
        actor.destroy()
        print('Process finished, actors removed.')