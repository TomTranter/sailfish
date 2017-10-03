<%!
    from sailfish import sym
%>

%if 'gravity' in context.keys():
	${const_var} float gravity = ${gravity}f;
%endif

<%def name="bgk_args_decl()">
	float rho, float c, float *iv0
</%def>

<%def name="bgk_args()">
	g0m0, g1m0, v
</%def>

%if subgrid != 'les-smagorinsky':
	${const_var} float tau0 = ${tau}f; // relaxation time
%endif
${const_var} float tau_ad = ${tau_ad}f; // relaxation time for the advection-diffusion
${const_var} float visc = ${visc}f;       // viscosity
${const_var} float visc_ad = ${visc_ad}f; // viscosity for the advection-diffusion
${const_var} float source_intensity = ${ad_source_intensity}f; // source intensity for the advection-diffusion

<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
${kernel_common.nonlocal_fields_decl()}
${kernel_common.body(bgk_args_decl)}

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>
<%namespace file="propagation.mako" import="*"/>

<%include file="tracers.mako"/>

<%def name="init_velocity_dist_with_eq()">
	%for local_var in bgk_equilibrium_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
	%endfor

	%for i, (feq, idx) in enumerate(bgk_equilibrium[0]):
		${get_odist('dist1_in', i)} = ${cex(feq, vectors=True)};
	%endfor
</%def>

<%def name="init_ad_dist_with_eq()">
	%for i, (feq, idx) in enumerate(bgk_equilibrium_ad[0]):
		${get_odist('dist2_in', i)} = ${cex(feq, vectors=True)};
	%endfor
</%def>

<%def name="init_dist_with_eq()">
	${init_velocity_dist_with_eq()}
	${init_ad_dist_with_eq()}
</%def>

%if dim == 2:
${kernel} void SetLocalVelocity(
	${global_ptr} float *dist1_in,
	${global_ptr} float *dist2_in,
	${global_ptr} float *irho,
	${kernel_args_1st_moment('ov')}
	int x, int y, float vx, float vy)
{
	int gx = x + get_global_id(0) - get_local_size(1) / 2;
	int gy = y + get_global_id(1) - get_local_size(1) / 2;

	${wrap_coords()}

	int gi = gx + ${arr_nx}*gy;
	float rho = irho[gi];
	float v0[${dim}];

	v0[0] = vx;
	v0[1] = vy;

	${init_velocity_dist_with_eq()}

	ovx[gi] = vx;
	ovy[gi] = vy;
}
%endif

// A kernel to set the node distributions using the equilibrium distributions
// and the macroscopic fields.
${kernel} void SetInitialConditions(
	${global_ptr} float *dist1_in,
	${global_ptr} float *dist2_in,
	${kernel_args_1st_moment('iv')}
	${global_ptr} float *irho,
	${global_ptr} float *ic)
{
	${local_indices()}

	// Cache macroscopic fields in local variables.
	float rho = irho[gi];
	float c = ic[gi];
	float v0[${dim}];

	v0[0] = ivx[gi];
	v0[1] = ivy[gi];
	%if dim == 3:
		v0[2] = ivz[gi];
	%endif

	${init_dist_with_eq()}
}

${kernel} void PrepareMacroFields(
	${global_ptr} int *map,
	${global_ptr} float *dist1_in,
	${global_ptr} float *dist2_in,
	${global_ptr} float *orho,
	${global_ptr} float *oc)
{
	${local_indices()}

	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type))
		return;

	int orientation = decodeNodeOrientation(ncode);

	Dist fi;
	float out;

	getDist(&fi, dist1_in, gi);
	get0thMoment(&fi, type, orientation, &out);
	orho[gi] = out;

	getDist(&fi, dist2_in, gi);
	get0thMoment(&fi, type, orientation, &out);
	oc[gi] = out;
}

${device_func} inline void BGK_relaxate_ad(${bgk_args_decl()}, Dist *d1, int node_type, int ncode)
{
	Dist feq1;
	float v0[${dim}];

	<%
		igrid = 0
	%>

	%for j in range(0, dim):
		%if igrid in force_for_eq:
			v0[${j}] = iv0[${j}] + ${cex(0.5 * sym.fluid_accel(sim, force_for_eq[igrid], j, forces, force_couplings), vectors=True)};
		%else:
			v0[${j}] = iv0[${j}] + ${cex(0.5 * sym.fluid_accel(sim, igrid, j, forces, force_couplings), vectors=True)};
		%endif
	%endfor

	%for feq, idx in bgk_equilibrium_ad[0]:
		feq1.${idx} = ${cex(feq, vectors=True)};
	%endfor

	%for idx in grid.idx_name:
		d1->${idx} += (feq1.${idx} - d1->${idx}) / tau_ad;
	%endfor
}

<%def name="relaxate_ad(bgk_args)">
	if (isWetNode(type)) {
		BGK_relaxate_ad(${bgk_args()}, &d1, type, ncode);
	}
</%def>


${kernel} void CollideAndPropagate(
	${global_ptr} int *map,
	${global_ptr} float *dist1_in,
	${global_ptr} float *dist1_out,
	${global_ptr} float *dist2_in,
	${global_ptr} float *dist2_out,
	${global_ptr} float *orho,
	${global_ptr} float *oc,
	${kernel_args_1st_moment('ov')}
	int save_macro)
{
	${local_indices()}

	// Shared variables for in-block propagation
	%for i in sym.get_prop_dists(grid, 1):
		${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
	%endfor
	%for i in sym.get_prop_dists(grid, 1):
		#define prop_${grid.idx_name[grid.idx_opposite[i]]} prop_${grid.idx_name[i]}
	%endfor

	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type))
		return;

	int orientation = decodeNodeOrientation(ncode);

	// Cache the distributions in local variables
	Dist d0, d1;
	getDist(&d0, dist1_in, gi);
	getDist(&d1, dist2_in, gi);

	// Macroscopic quantities for the current cell
	float g0m0, v[${dim}], g1m0;

	getMacro(&d0, ncode, type, orientation, &g0m0, v);
	get0thMoment(&d1, type, orientation, &g1m0);

	precollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v);
	precollisionBoundaryConditions(&d1, ncode, type, orientation, &g1m0, v);
	${relaxate(bgk_args)}
	${relaxate_ad(bgk_args)}
	postcollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v, gi, dist1_out);
	postcollisionBoundaryConditions(&d1, ncode, type, orientation, &g1m0, v, gi, dist2_out);

	// Source for advection-diffusion
	if (gx >= ${ad_source_rect_left} && gx < ${ad_source_rect_right} && gy >= ${ad_source_rect_bottom} && gy < ${ad_source_rect_top})
	{
		%for i, idx in enumerate(grid.idx_name):
			d1.${idx} += (float)${grid.weights[i]}*source_intensity;
		%endfor
	}

	// only save the macroscopic quantities if requested to do so
	if (save_macro == 1) {
		orho[gi] = g0m0;
		ovx[gi] = v[0];
		ovy[gi] = v[1];
		%if dim == 3:
			ovz[gi] = v[2];
		%endif
		oc[gi] = g1m0;
	}

	${propagate('dist1_out', 'd0')}
	${barrier()}
	${propagate('dist2_out', 'd1')}
}
