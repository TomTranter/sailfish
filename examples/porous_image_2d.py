#!/usr/bin/env python -u
"""Demonstrates how to load geometry from an external file.

The external file is a Boolean numpy array in the .npy format.
Nodes marked as True indicate walls.

In order to generate a .npy file from an STL geometry, check
out utils/voxelizer.

The sample file pipe.npy was generated using:
  a = np.zeros((128, 41, 41), dtype=np.bool)
  hz, hy, hx = np.mgrid[0:41, 0:41, 0:128]
  a[(hz - 20)**2 + (hy - 20)**2 >
    (19.3 * (0.8 + 0.2 * np.sin(2 * pi * hx / 128.0)))**2] = True
"""

import numpy as np
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim
import time
import csv
from scipy.misc import imread
# No. neighbor interactions
mypath='/home/tom/test_images/ml/Testing'
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
file=None
all_perm = []
all_tau = []

def save_data(data=None):
    with open('sailfish_data.csv', 'a', newline='\n') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(data)
# File Path
save_data([mypath])
# Header
save_data(['file', 'porosity','perm', 'tau', 'iters', 'convergence'])

class ExternalSubdomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0

    def boundary_conditions(self, hx, hy):
        if hasattr(self.config, '_wall_map'):
            partial_wall_map = self.select_subdomain(
                self.config._wall_map, hx, hy)
            self.set_node(partial_wall_map, NTFullBBWall)

    # Only used with node_addressing = 'indirect'.
    def load_active_node_map(self, hx, hy):
        partial_wall_map = self.select_subdomain(
            self.config._wall_map, hx, hy)
        self.set_active_node_map_from_wall_map(partial_wall_map)


class ExternalSimulation(LBFluidSim, LBForcedSim):
    subdomain = ExternalSubdomain
    data_table = []
    F = 1e-5
    mu = 8.9e-4
    phi = 1.0

    def read_jpeg(_file):
        im_rgb = imread(_file)
        im_r = im_rgb[:, :, 0]
        solid = np.unique(im_r)[0]
        return im_r == solid
    
    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--geometry', type=str, default='pipe.npy',
                           help='file defining the geometry')

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'visc': cls.mu,
            'grid': 'D2Q9',
            'max_iters': 100000,
            'every': 250,
            'periodic_x': True})

    @classmethod
    def modify_config(cls, config):
        global file
        if not config.geometry:
            return
        try:
            config.geometry = np.load(file)
        except:
            try:
                config.geometry = cls.read_jpeg(file)
            except:
                print("FILE TYPE NOT VALID")
        # Override lattice size based on the geometry file.
        wall_map = ~config.geometry.astype(bool)
        cls.phi = 1-(np.sum(wall_map)/np.size(wall_map))
        config.lat_ny, config.lat_nx = wall_map.shape
        # Add nodes corresponding to ghosts. Assumes an envelope size of 1.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map
    
    def _u_mag(self, runner):
        m = runner._output._fluid_map
        u_mag = np.sqrt(runner._sim.vx[m]**2 + runner._sim.vy[m]**2)
        return u_mag

    def _u_norm(self, runner):
        #m = runner._output._fluid_map
        u = self._u_mag(runner)
        normu = np.linalg.norm(u) / u.size
        return normu
    
    def _calc_perm(self, runner):
        m = runner._output._fluid_map
        mean_u = np.mean(runner._sim.vx[m])
        perm = (self.mu * mean_u * self.phi)/self.F
        return perm
    
    def _calc_tau(self, runner):
        u_mag = self._u_mag(runner)
        m = runner._output._fluid_map
        tau = np.sum(u_mag)/np.sum(runner._sim.vx[m])
        return tau
        
    def after_step(self, runner):
        global file
        every_n = 250
        
        # Request the velocity field one step before actual processing.
        if self.iteration % every_n == every_n - 1:
            self.need_sync_flag = True

        # Calculate and save the norm of valocity field.
        if self.iteration == every_n:
            self.u_old = self._u_norm(runner)
#            from pprint import pprint
#            print('Simulation Variables')
#            pprint(vars(runner._sim).keys())
#            for fo in runner._sim.force_objects:
#                print(fo.force())

        if self.iteration % every_n == 0 and self.iteration > every_n:
            u_norm = self._u_norm(runner)
            du_norm = u_norm - self.u_old
            self.convergence = np.abs(du_norm/u_norm)
            self.u_old = u_norm
            perm = self._calc_perm(runner)
            tau = self._calc_tau(runner)
            print('permeability', perm, 'tau', tau)
            print('convergence', self.convergence)
            self.data_table.append((self.iteration, du_norm, u_norm, perm, tau))
            # Crude convergence tactic
            if self.convergence < 1e-5:
                # Stop running on next iteration
                # May be a better way to do this !!!
                runner.config.max_iters = self.iteration + 1

        if self.iteration == self.config.max_iters - 1:
#            data_table_np = np.array(self.data_table)
#            np.savez('unorm',
#                     it=data_table_np[:, 0],
#                     du_norm=data_table_np[:, 1],
#                     u_norm=data_table_np[:, 2],
#                     perm=data_table_np[:, 3],
#                     tau=data_table_np[:, 4])
            print('SAVING DATA')
            save_data([file.split('/')[-1].split('.')[0],
                       str(self.phi),
                       str(self._calc_perm(runner)),
                       str(self._calc_tau(runner)),
                       str(self.iteration),
                       str(self.convergence)])
            

    def __init__(self, config):
        super(ExternalSimulation, self).__init__(config)
        self.add_body_force((self.F, 0.0))


if __name__ == '__main__':
    for f in onlyfiles:
        file= join(mypath, f)
        print('Processing', file)
        ctrl = LBSimulationController(ExternalSimulation)
        st = time.time()
        ctrl.run()
        print('Sim time:', time.time()-st)