#!/usr/bin/python -u

import numpy
from examples import ldc_2d
import optparse
import sys
from sailfish import lb_single, geo, sym
from sailfish.controller import LBSimulationController


class LBAdvectionDiffusion(lb_single.LBFluidSim):
    filename    = 'advection_diffusion'
    kernel_file = 'advection_diffusion.mako'
    

    #def __init__(self, geo_class, options=[], args=None, defaults=None):
    #def __init__(self):
        #super(LBAdvectionDiffusion, self).__init__(geo_class, options, args, defaults)
        #super(LBAdvectionDiffusion, self).__init__(geo_class)
    def __init__(self, config):
        super(LBAdvectionDiffusion, self).__init__(config)
        self._prepare_symbols()
        #self.add_nonlocal_field(0)
        #self.add_nonlocal_field(1)
        
        self.equilibrium, self.equilibrium_vars = sym.bgk_equilibrium(self.grid, self.S.rho, self.S.rho)
        self.equilibrium_ad, _ = sym.bgk_equilibrium(self.grid, self.S.c, self.S.c)
        
    def _add_options(self, parser, lb_group):
        super(LBAdvectionDiffusion, self)._add_options(parser, lb_group)

        lb_group.add_option('--visc_ad', dest='visc_ad', help='viscosity for the advection-diffusion', action='store', type='float', default=0.01)
        
        # Source term of the advection-diffusion equation is restricted to a rectangle
        lb_group.add_option('--ad_source_intensity', dest='ad_source_intensity', help='advection-diffusion source intensity', action='store', type='float', default=0.0)
        lb_group.add_option('--ad_source_rect_left', dest='ad_source_rect_left', help='advection-diffusion source rectangle left coordinate', action='store', type='float', default=0.0)
        lb_group.add_option('--ad_source_rect_right', dest='ad_source_rect_right', help='advection-diffusion source rectangle right coordinate', action='store', type='float', default=0.0)
        lb_group.add_option('--ad_source_rect_top', dest='ad_source_rect_top', help='advection-diffusion source rectangle top coordinate', action='store', type='float', default=0.0)
        lb_group.add_option('--ad_source_rect_bottom', dest='ad_source_rect_bottom', help='advection-diffusion source rectangle bottom coordinate', action='store', type='float', default=0.0)

        return None
        
    def _update_ctx(self, ctx):
        super(LBAdvectionDiffusion, self)._update_ctx(ctx)
        ctx['visc_ad'] = self.options.visc_ad
        ctx['tau_ad'] = self.float((6.0 * self.options.visc_ad + 1.0)/2.0)
        ctx['bgk_equilibrium_ad'] = self.equilibrium_ad
        ctx['ad_source_intensity'] = self.options.ad_source_intensity*self.dt
        ctx['ad_source_rect_left'] = self.options.ad_source_rect_left
        ctx['ad_source_rect_right'] = self.options.ad_source_rect_right
        ctx['ad_source_rect_top'] = self.options.ad_source_rect_top
        ctx['ad_source_rect_bottom'] = self.options.ad_source_rect_bottom
        
    @property
    def sim_info(self):
        ret = lb_single.LBFluidSim.sim_info.fget(self)
        ret['visc ad'] = self.options.visc_ad
        return ret
        
    def _prepare_symbols(self):
        self.S.alias('c', self.S.g1m0) # Concentration
        
    def _init_fields(self, need_dist):
        super(LBAdvectionDiffusion, self)._init_fields(need_dist)
        self.c = self.make_field('concentration', True)

        if need_dist:
            self.dist2 = self.make_dist(self.grid)
            
        self.vis.add_field(self.c, 'concentration')
        
    def curr_dists(self):
        if self.iter_ & 1:
            return [self.gpu_dist1b, self.gpu_dist2b]
        else:
            return [self.gpu_dist1a, self.gpu_dist2a]
            
    def _init_compute_fields(self):
        super(LBAdvectionDiffusion, self)._init_compute_fields()
        self.gpu_c = self.backend.alloc_buf(like=self.c)
        self.gpu_mom0.append(self.gpu_c)

        if not self._ic_fields:
            self.gpu_dist2a = self.backend.alloc_buf(like=self.dist2)
            self.gpu_dist2b = self.backend.alloc_buf(like=self.dist2)
        else:
            self.gpu_dist2a = self.backend.alloc_buf(size=self.get_dist_bytes(self.grid), wrap_in_array=False)
            self.gpu_dist2b = self.backend.alloc_buf(size=self.get_dist_bytes(self.grid), wrap_in_array=False)

        self.img_rho = self.bind_nonlocal_field(self.gpu_rho, 0)
        self.img_c = self.bind_nonlocal_field(self.gpu_c, 1)
        
    def _init_compute_kernels(self):
        cnp_args1n = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_dist2a, self.gpu_dist2b, self.gpu_rho, self.gpu_c] + self.gpu_velocity + [numpy.uint32(0)]
        cnp_args1s = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_dist2a, self.gpu_dist2b, self.gpu_rho, self.gpu_c] + self.gpu_velocity + [numpy.uint32(1)]
        cnp_args2n = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_dist2b, self.gpu_dist2a, self.gpu_rho, self.gpu_c] + self.gpu_velocity + [numpy.uint32(0)]
        cnp_args2s = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_dist2b, self.gpu_dist2a, self.gpu_rho, self.gpu_c] + self.gpu_velocity + [numpy.uint32(1)]

        macro_args1 = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist2a, self.gpu_rho, self.gpu_c]
        macro_args2 = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist2b, self.gpu_rho, self.gpu_c]

        k_block_size = self._kernel_block_size()
        cnp_name = 'CollideAndPropagate'
        macro_name = 'PrepareMacroFields'
        fields = [self.img_rho, self.img_c]

        kern_cnp1n = self.backend.get_kernel(self.mod, cnp_name, args=cnp_args1n, args_format='P'*(len(cnp_args1n)-1)+'i', block=k_block_size, fields=fields)
        kern_cnp1s = self.backend.get_kernel(self.mod, cnp_name, args=cnp_args1s, args_format='P'*(len(cnp_args1n)-1)+'i', block=k_block_size, fields=fields)
        kern_cnp2n = self.backend.get_kernel(self.mod, cnp_name, args=cnp_args2n, args_format='P'*(len(cnp_args1n)-1)+'i', block=k_block_size, fields=fields)
        kern_cnp2s = self.backend.get_kernel(self.mod, cnp_name, args=cnp_args2s, args_format='P'*(len(cnp_args1n)-1)+'i', block=k_block_size, fields=fields)
        
        kern_mac1 = self.backend.get_kernel(self.mod, macro_name, args=macro_args1, args_format='P'*len(macro_args1), block=k_block_size)
        kern_mac2 = self.backend.get_kernel(self.mod, macro_name, args=macro_args2, args_format='P'*len(macro_args2), block=k_block_size)

        # For occupancy analysis in performance tests.
        self._lb_kernel = kern_cnp1n

        # Map: iteration parity -> kernel arguments to use.
        self.kern_map = {
            0: (kern_mac1, kern_cnp1n, kern_cnp1s),
            1: (kern_mac2, kern_cnp2n, kern_cnp2s),
        }

        if self.grid.dim == 2:
            self.kern_grid_size = (self.arr_nx/self.options.block_size, self.arr_ny)
        else:
            self.kern_grid_size = (self.arr_nx/self.options.block_size * self.arr_ny, self.arr_nz)
            
    def _init_compute_ic(self):
        if not self._ic_fields:
            # Nothing to do, the initial distributions have already been
            # set and copied to the GPU in _init_compute_fields.
            return

        args1 = [self.gpu_dist1a, self.gpu_dist2a] + self.gpu_velocity + [self.gpu_rho, self.gpu_c]
        args2 = [self.gpu_dist1b, self.gpu_dist2b] + self.gpu_velocity + [self.gpu_rho, self.gpu_c]

        kern1 = self.backend.get_kernel(self.mod, 'SetInitialConditions', args=args1, args_format='P'*len(args1), block=self._kernel_block_size())
        kern2 = self.backend.get_kernel(self.mod, 'SetInitialConditions', args=args2, args_format='P'*len(args2), block=self._kernel_block_size())

        self.backend.run_kernel(kern1, self.kern_grid_size)
        self.backend.run_kernel(kern2, self.kern_grid_size)
        self.backend.sync()
        
    def _lbm_step(self, get_data, **kwargs):
        kerns = self.kern_map[self.iter_ & 1]

        self.backend.run_kernel(kerns[0], self.kern_grid_size)
        self.backend.sync()

        if get_data:
            self.backend.run_kernel(kerns[2], self.kern_grid_size)
            self.backend.sync()
            self.hostsync_velocity()
            self.hostsync_density()
            self.backend.from_buf(self.gpu_c)
            self.backend.sync()
        else:
            self.backend.run_kernel(kerns[1], self.kern_grid_size)


class LBMGeoLDC_AD(ldc_2d.LDCBlock):
    max_v = 0.1
    
    def init_fields(self):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]
        self.sim.rho[:] = 1.0
        self.sim.vx[hy == self.lat_ny-1] = self.max_v
        
        # c stands for concentration - the quantity which evolves according to the advection-diffusion equation
        self.sim.c[:] = 0.0

        
class LDCSim_AD(LBAdvectionDiffusion):
    subdomain = LBMGeoLDC_AD
    filename = 'ldc_ad'
    
    #def __init__(self, geo_class):
    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'model': 'bgk',
            #'bc_velocity': 'equilibrium',
            'verbose': True,
            #'visc_ad': 0.01,
            #'ad_source_intensity': 0.1,
            #'ad_source_rect_left': 20.0,
            #'ad_source_rect_right': 30.0,
            #'ad_source_rect_top': 70.0,
            #'ad_source_rect_bottom': 60.0,
            'grid': 'D2Q9'
            })

        #settings = {'model': 'bgk', 'bc_velocity': 'equilibrium', 'verbose': True, 'visc_ad': 0.01, 'ad_source_intensity': 0.1, 'ad_source_rect_left': 20, 'ad_source_rect_right': 30, 'ad_source_rect_top': 70, 'ad_source_rect_bottom': 60}
        #settings.update(defaults)
        #super(LDCSim_AD, self).__init__(geo_class, defaults=settings)
        #super(LDCSim_AD, self).__init__(geo_class)


if __name__ == '__main__':
    sim = LBSimulationController(LDCSim_AD)
    sim.run()
