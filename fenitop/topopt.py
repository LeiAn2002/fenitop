"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import time

import numpy as np
from mpi4py import MPI
from dolfinx.io import XDMFFile

from fenitop.fem import form_fem
from fenitop.parameterize import DensityFilter, Heaviside, Normalization
from fenitop.sensitivity import Sensitivity
from fenitop.optimize import optimality_criteria, mma_optimizer
from fenitop.utility import Communicator, Plotter, save_xdmf
from fenitop.sensitivity_ksi import Sensitivity_ksi
from fenitop.check import sensitivity_check
from fenitop.update_c import FieldUpdater
# from fenitop.check_dcdksi import Sensitivity_check_cksi


def topopt(fem, opt):
    """Main function for topology optimization."""

    # Initialization
    comm = MPI.COMM_WORLD
    block_types = opt["block_types"]
    
    check_sens = False
    linear_problem, u_field, lambda_field, rho_field, rho_phys_field, ksi_field_list, ksi_phys_field_list, c_field_list, local_vf_field, local_vf_phys_field = form_fem(fem, opt)
    density_filter_rho = DensityFilter(comm, rho_field, rho_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])
    heaviside_rho = Heaviside(rho_phys_field)

    # 注意，这里因为我犯懒，所以dVdvf写在了Sensitivity里，而dCdvf和dUdvf写在了Sensitivity_ksi里
    sens_problem = Sensitivity(comm, opt, linear_problem, u_field, lambda_field, rho_phys_field, local_vf_phys_field)
    sens_problem_ksi_and_vf = Sensitivity_ksi(opt, linear_problem, u_field, lambda_field, ksi_phys_field_list, c_field_list, local_vf_phys_field)
    S_comm = Communicator(rho_phys_field.function_space, fem["mesh_serial"])
    c_update = FieldUpdater(opt, ksi_phys_field_list, c_field_list, local_vf_phys_field)  # if parallel, maybe need to use comm

    density_filter_ksi_list = []
    # heaviside_ksi_list = []
    # comm_ksi_list = []

    for i in range(block_types):
        density_filter_ksi_list.append(DensityFilter(comm, ksi_field_list[i], ksi_phys_field_list[i],
                                                opt["filter_radius"], fem["petsc_options"]))
        # heaviside_ksi_list.append(Heaviside(ksi_phys_field_list[i]))

        # comm_ksi_list.append(Communicator(ksi_phys_field_list[i].function_space, fem["mesh_serial"]))
    normal_ksi = Normalization(ksi_phys_field_list)

    density_filter_vf = DensityFilter(comm, local_vf_field, local_vf_phys_field,
                                   opt["filter_radius"], fem["petsc_options"])
    # heaviside_vf = Heaviside(local_vf_phys_field)

    if comm.rank == 0:
        plotter = Plotter(fem["mesh_serial"])
    num_consts = 1 if opt["opt_compliance"] else 2
    # num_consts = 1
    num_elems = rho_field.vector.array.size
    num_checked_elems = 20
    eps = 1e-2

    if not opt["use_oc"]:
        theta_old1, theta_old2 = np.zeros(num_elems*4), np.zeros(num_elems*4)
        low, upp = None, None

    low, upp = None, None

    # Apply passive zones

    centers = rho_field.function_space.tabulate_dof_coordinates()[:num_elems].T
    solid, void = opt["solid_zone"](centers), opt["void_zone"](centers)
    rho_ini = np.full(num_elems, opt["vol_frac"])
    rho_ini[solid], rho_ini[void] = 0.995, 0.005
    rho_field.vector.array[:] = rho_ini
    ksi_ini = np.full(num_elems, 1/3)
    for i in range(block_types):
        ksi_field_list[i].vector.array[:] = ksi_ini
    vf_ini = np.full(num_elems, 0.5)
    local_vf_field.vector.array[:] = vf_ini
    rho_min, rho_max = np.zeros(num_elems), np.ones(num_elems)
    ksi_min, ksi_max = 0.1*np.ones(num_elems), 0.9*np.ones(num_elems)
    local_vf_min, local_vf_max = 0.2*np.ones(num_elems), 0.7*np.ones(num_elems)
    theta_min = np.concatenate((rho_min, np.tile(ksi_min, block_types), local_vf_min))
    theta_max = np.concatenate((rho_max, np.tile(ksi_max, block_types), local_vf_max))
    rho_min[solid], rho_max[void] = 0.99, 0.01

    # Start topology optimization
    opt_iter, beta, change = 0, 1, 2*opt["opt_tol"]
    while opt_iter < opt["max_iter"] and change > opt["opt_tol"]:
        opt_start_time = time.perf_counter()
        opt_iter += 1

        # Density filter and Heaviside projection
        
        density_filter_rho.forward()

        for i in range(block_types):
            density_filter_ksi_list[i].forward()

        density_filter_vf.forward()

        if opt_iter % opt["beta_interval"] == 0 and beta < opt["beta_max"]:
            beta *= 2
            change = opt["opt_tol"] * 2

        heaviside_rho.forward(beta)

        # for i in range(block_types):
        #     heaviside_ksi_list[i].forward(beta)

        # heaviside_vf.forward(beta)

        normal_ksi.forward()

        c_update.update()
        
        # Solve FEM
        linear_problem.solve_fem()

        # Compute function values and sensitivities
        [C_value, V_value, U_value], sensitivities, dVdvf_vec = sens_problem.evaluate()
       
        heaviside_rho.backward(sensitivities)

        [dCdrho, dVdrho, dUdrho] = density_filter_rho.backward(sensitivities)
        # print(max(np.array(dVdvf_vec)))

        dVdvf = density_filter_vf.backward([dVdvf_vec])[0]
        # print(max(np.array(dVdvf)))

        sensitivities_ksi_and_vf, dCdksi_vector_ksi_and_vf = sens_problem_ksi_and_vf.evaluate()

        dUdksi_middle_list = []
        dCdksi_middle_list = []
        
        # 下面这段是在干啥：sensitivity需要多次求和（参见CMAME）
        if not opt["opt_compliance"]:
            for j in range(block_types):
                res_dUdksi_middle = sensitivities_ksi_and_vf[0][0].duplicate()
                res_dUdksi_middle.zeroEntries()
                for k in range(block_types):
                    res_dUdksi_middle_part2 = sensitivities_ksi_and_vf[0][0].duplicate()
                    res_dUdksi_middle_part2.zeroEntries()
                    for i in range(6):
                        res_dUdksi_middle_part2.axpy(1.0, sensitivities_ksi_and_vf[k][i])
                    normal_ksi.backward(res_dUdksi_middle_part2, j, k)

                    res_dUdksi_middle.axpy(1.0, res_dUdksi_middle_part2)

                dUdksi_middle_list.append(res_dUdksi_middle)

            dUdksi_list = []
            for i in range(block_types):
                dUdksii = density_filter_ksi_list[i].backward([dUdksi_middle_list[i]])
                dUdksi_list.append(dUdksii[0])

        for j in range(block_types):
            res_dCdksi_middle = dCdksi_vector_ksi_and_vf[0][0].duplicate()
            res_dCdksi_middle.zeroEntries()
            for k in range(block_types):
                res_dCdksi_middle_part2 = dCdksi_vector_ksi_and_vf[0][0].duplicate()
                res_dCdksi_middle_part2.zeroEntries()

                for i in range(6):
                    res_dCdksi_middle_part2.axpy(1.0, dCdksi_vector_ksi_and_vf[k][i])

                normal_ksi.backward(res_dCdksi_middle_part2, j, k)

                res_dCdksi_middle.axpy(1.0, res_dCdksi_middle_part2)

            dCdksi_middle_list.append(res_dCdksi_middle)

        # print(np.linalg.norm(dCdksi_middle_list[0].array-dCdksi_middle_list[1].array))
        # print(np.linalg.norm(dCdksi_vector_list[0][0].array-dCdksi_vector_list[1][0].array))

        dCdksi_list = []

        for i in range(block_types):
            dCdksii = density_filter_ksi_list[i].backward([dCdksi_middle_list[i]])
            dCdksi_list.append(dCdksii[0])
        
        if not opt["opt_compliance"]:
            res_dUdkvf_middle = sensitivities_ksi_and_vf[0][0].duplicate()
            res_dUdkvf_middle.zeroEntries()
            for i in range(6):
                res_dUdkvf_middle.axpy(1.0, sensitivities_ksi_and_vf[block_types][i])
            dUdvf = density_filter_vf.backward([res_dUdkvf_middle])[0]

        res_dCdvf_middle = dCdksi_vector_ksi_and_vf[0][0].duplicate()
        res_dCdvf_middle.zeroEntries()
        for i in range(6):
            res_dCdvf_middle.axpy(1.0, dCdksi_vector_ksi_and_vf[block_types][i])
        dCdvf = density_filter_vf.backward([res_dCdvf_middle])[0]

        if opt["opt_compliance"]:
            g_vec = np.array([V_value-opt["vol_frac"]])
            dJdrho, dgdrho = dCdrho, np.vstack([dVdrho])
            zero_array = np.zeros_like(dVdrho)
            dVdtheta = np.concatenate((dVdrho, np.tile(zero_array, block_types), dVdvf))
            dgdtheta = np.vstack([dVdtheta])
            dgdvf = np.vstack([dVdvf])
            dJdtheta = np.concatenate([dCdrho] + dCdksi_list + [dCdvf])

        else:
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            dCdtheta = np.concatenate([dCdrho] + dCdksi_list + [dCdvf])
            zero_array = np.zeros_like(dVdrho)
            dVdtheta = np.concatenate((dVdrho, np.tile(zero_array, block_types), dVdvf))
            # dgdrho = np.vstack([dVdrho, dCdrho])
            # dgdksi_1 = np.vstack([zero_array, dCdksi_list[0]])
            dgdvf = np.vstack([dVdvf, dCdvf])
            dJdrho, dgdtheta = dUdrho, np.vstack([dVdtheta, dCdtheta])
            dJdtheta = np.concatenate([dJdrho] + dUdksi_list + [dUdvf])

        # if check_sens:
        #     sensitivity_check(
        #         comm, opt, linear_problem, sens_problem, rho_field, num_elems, num_consts,
        #         num_checked_elems, eps, U_value, g_vec, dJdrho, dgdrho,
        #         density_filter_rho, heaviside, density_filter_ksi_list, normal_ksi, beta, c_update)

        if check_sens:
            if opt["opt_compliance"]:
                sensitivity_check(
                    comm, opt, linear_problem, sens_problem, local_vf_field, num_elems, num_consts,
                    num_checked_elems, eps, C_value, g_vec, dCdvf, dgdvf,
                    density_filter_rho, heaviside_rho, density_filter_ksi_list, normal_ksi, density_filter_vf, beta, c_update)
            else:
                sensitivity_check(
                    comm, opt, linear_problem, sens_problem, local_vf_field, num_elems, num_consts,
                    num_checked_elems, eps, U_value, g_vec, dUdvf, dgdvf,
                    density_filter_rho, heaviside_rho, density_filter_ksi_list, normal_ksi, density_filter_vf, beta, c_update)
        # if check_sens:
        #     if opt["opt_compliance"]:
        #         sensitivity_check(
        #             comm, opt, linear_problem, sens_problem, ksi_field_list[0], num_elems, num_consts,
        #             num_checked_elems, eps, C_value, g_vec, dCdksi_list[0], dgdvf,
        #             density_filter_rho, heaviside_rho, density_filter_ksi_list, normal_ksi, density_filter_vf, beta, c_update)
        #     else:
        #         sensitivity_check(
        #             comm, opt, linear_problem, sens_problem, ksi_field_list[0], num_elems, num_consts,
        #             num_checked_elems, eps, U_value, g_vec, dUdksi_list[0], dgdvf,
        #             density_filter_rho, heaviside_rho, density_filter_ksi_list, normal_ksi, density_filter_vf, beta, c_update)
            
            # if opt["opt_compliance"]:
            #     sensitivity_check(
            #         comm, opt, linear_problem, sens_problem, rho_field, num_elems, num_consts,
            #         num_checked_elems, eps, C_value, g_vec, dCdrho, dgdrho,
            #         density_filter_rho, heaviside, density_filter_ksi_list, normal_ksi, beta, c_update)
            # else:
            #     sensitivity_check(
            #         comm, opt, linear_problem, sens_problem, ksi_field_list[0], num_elems, num_consts,
            #         num_checked_elems, eps, U_value, g_vec, dUdksi_list[0], dgdksi_1,
            #         density_filter_rho, heaviside, density_filter_ksi_list, normal_ksi, beta, c_update)
            
        # Update the design variables
        rho_values = rho_field.vector.array.copy()
        ksi_value_list = []
        for i in range(block_types):
            ksi_value_list.append(ksi_field_list[i].vector.array.copy())
        vf_values = local_vf_field.vector.array.copy()
        theta_values = np.concatenate([rho_values] + ksi_value_list + [vf_values])
        if opt["opt_compliance"] and opt["use_oc"]:
            # rho_new, change = optimality_criteria(
            #     rho_values, rho_min, rho_max, g_vec, dJdrho, dgdrho[0], opt["move"])
            theta_new, change = optimality_criteria(
                theta_values, theta_min, theta_max, g_vec, dJdtheta, dgdtheta[0], opt["move"])

        else:
            theta_new, change, low, upp = mma_optimizer(
                num_consts, num_elems*(block_types+2), opt_iter, theta_values, theta_min, theta_max,
                theta_old1, theta_old2, dJdtheta, g_vec, dgdtheta, low, upp)
            theta_old2 = theta_old1.copy()
            theta_old1 = theta_values.copy()

        rho_field.vector.array = theta_new.copy()[0:num_elems]
        
        for i in range(block_types):
            j = i+1
            ksi_field_list[i].vector.array = theta_new.copy()[j*num_elems:(j+1)*num_elems]

        j += 1
        local_vf_field.vector.array = theta_new.copy()[j*num_elems:(j+2)*num_elems]
        
        # Output the histories
        opt_time = time.perf_counter() - opt_start_time
        if comm.rank == 0:
            print(f"opt_iter: {opt_iter}, opt_time: {opt_time:.3g} (s), "
                  f"beta: {beta}, C: {C_value:.3f}, V: {V_value:.3f}, "
                  f"U: {U_value:.3f}, change: {change:.3f}", flush=True)
        
    values = S_comm.gather(rho_phys_field)
    if comm.rank == 0:
        plotter.plot(values)

    save_xdmf(fem["mesh"], rho_phys_field, "/shared/fenitop_for_virtualgrowth/data/rho_field.xdmf")

    with XDMFFile(fem["mesh"].comm, "/shared/fenitop_for_virtualgrowth/data/ksi_field_1.xdmf", "w") as xdmf:
        xdmf.write_mesh(fem["mesh"]) 
        xdmf.write_function(ksi_phys_field_list[0]) 

    with XDMFFile(fem["mesh"].comm, "/shared/fenitop_for_virtualgrowth/data/ksi_field_2.xdmf", "w") as xdmf:
        xdmf.write_mesh(fem["mesh"]) 
        xdmf.write_function(ksi_phys_field_list[1]) 

    with XDMFFile(fem["mesh"].comm, "/shared/fenitop_for_virtualgrowth/data/ksi_field_3.xdmf", "w") as xdmf:
        xdmf.write_mesh(fem["mesh"]) 
        xdmf.write_function(ksi_phys_field_list[2]) 

    with XDMFFile(fem["mesh"].comm, "/shared/fenitop_for_virtualgrowth/data/vf_field.xdmf", "w") as xdmf:
        xdmf.write_mesh(fem["mesh"]) 
        xdmf.write_function(local_vf_phys_field) 