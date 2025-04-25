import random
import time

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def sensitivity_check(
        comm, opt, linear_problem, sens_problem, rho_field, num_elems, num_consts,
        num_checked_elems, eps, J_old, g_vec_old, dJdrho_formula, dgdrho_formula,
        density_filter, heaviside, density_filter_ksi_list, normal_ksi, density_filter_vf, beta, c_update):
    """Perform the sensitivity check."""

    # Initialization
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if rank == 0:
        print("Start the sensitivity check.", flush=True)

    rank_tags = np.full(num_elems, rank)
    element_tags = np.arange(num_elems)
    global_rank_tags = comm.gather(rank_tags, root=0)
    global_element_tags = comm.gather(element_tags, root=0)
    global_num_elems = comm.allreduce(num_elems, op=MPI.SUM)

    global_dJdrho_formula = comm.gather(dJdrho_formula, root=0)
    global_dgdrho_formula = comm.gather(dgdrho_formula, root=0)

    if rank == 0:
        num_checked_elems = min(global_num_elems, num_checked_elems)
        global_sample_elements = random.sample(range(global_num_elems), num_checked_elems)

        global_rank_tags = np.hstack(global_rank_tags)
        global_element_tags = np.hstack(global_element_tags)
        sample_ranks = global_rank_tags[global_sample_elements]
        sample_elements = global_element_tags[global_sample_elements]

        global_dJdrho_formula = np.hstack(global_dJdrho_formula)
        global_dgdrho_formula = np.hstack(global_dgdrho_formula)
        dJdrho_formula = global_dJdrho_formula[global_sample_elements]
        dgdrho_formula = global_dgdrho_formula[:, global_sample_elements]

        dJdrho_diff = np.zeros(num_checked_elems)
        dgdrho_diff = np.zeros((num_consts, num_checked_elems))
    else:
        sample_ranks = sample_elements = None
    sample_ranks = MPI.COMM_WORLD.bcast(sample_ranks, root=0)
    sample_elements = MPI.COMM_WORLD.bcast(sample_elements, root=0)

    # Start the sensitivity check
    rho_old = rho_field.vector.array.copy()
    for check_iter, (sample_rank, sample_element) in enumerate(zip(sample_ranks, sample_elements)):
        check_start_time = time.time()

        rho_new = rho_old.copy()
        if rank == sample_rank:
            rho_new[sample_element] += eps
        rho_field.vector.array = rho_new.copy()

        density_filter_vf.forward()

        # Density filter and heaviside projection
        density_filter.forward()
        heaviside.forward(beta)

        for i in range(3):
            density_filter_ksi_list[i].forward()
        normal_ksi.forward()

        c_update.update()

        # Solve FEM
        linear_problem.solve_fem()

        # Compute function values
        [C_value, V_value, U_value], sensitivities = sens_problem.evaluate()
        
        if opt["opt_compliance"]:
            J, g_vec = C_value, np.array([V_value-opt["vol_frac"]])
        else:
            J = U_value
            # J = c_field_list[0].x.array[0]
            g_vec = np.array([V_value-opt["vol_frac"], C_value-opt["compliance_bound"]])
            # g_vec = np.array([V_value-opt["vol_frac"]])

        if rank == 0:
            dJdrho_diff[check_iter] = (J-J_old) / eps
            dgdrho_diff[:, check_iter] = (g_vec-g_vec_old) / eps
            abs_err_dCdrho = abs(dJdrho_diff[check_iter] - dJdrho_formula[check_iter])
            abs_err_dgdrho = np.abs(dgdrho_diff[:, check_iter] - dgdrho_formula[:, check_iter])
            check_end_time = time.time()
            check_time = check_end_time - check_start_time
            # print(f"check_iter: {check_iter+1}, time: {check_time:.3g}(s), "
            #       f"check_rank: {sample_rank}, check_elem: {sample_element}, "
            #       f"J: {J:.6g}, J_old: {J_old:.6g}, "
            #       f"abs_err_dCdrho: {abs_err_dCdrho:.3g}, "
            #       f"abs_err_dgdrho: " + np.array2string(abs_err_dgdrho, formatter={"float": "{:.3g}".format}), flush=True)

    # Post-processing
    if rank == 0:
        abs_err_obj = np.linalg.norm(dJdrho_formula-dJdrho_diff, ord=2)
        rel_err_obj = abs_err_obj / np.linalg.norm(dJdrho_formula, ord=2)

        abs_err_const = np.linalg.norm(dgdrho_formula-dgdrho_diff, axis=1, ord=2)
        rel_err_const = abs_err_const / np.linalg.norm(dgdrho_formula, axis=1, ord=2)

        print(f"[obj_func] absolute 2-norm error: {abs_err_obj:.3g}\n"
              f"[obj_func] relative 2-norm error: {rel_err_obj:.3g}\n"
              f"[const_func(s)] absolute 2-norm error(s): "
              + np.array2string(abs_err_const, formatter={"float": "{:.3g}".format})
              + "\n[const_func(s)] relative 2-norm error(s): "
              + np.array2string(rel_err_const, formatter={"float": "{:.3g}".format}),
              flush=True)
        plot_sensitivity_check(dJdrho_formula, dJdrho_diff,
                               dgdrho_formula, dgdrho_diff)


def plot_sensitivity_check(dJdz_formula, dJdz_diff,
                           dgdz_formula, dgdz_diff, fig_path=""):
    """Plot the sensitivity check."""

    element_tags = np.arange(dJdz_formula.size) + 1
    fontsize = 15

    # Objective function
    plt.figure(figsize=(8, 6))
    sort_idx = np.argsort(dJdz_formula)
    plt.plot(element_tags, dJdz_diff[sort_idx], color="red", linestyle="-",
             lw=3, label="Finite difference")
    plt.plot(element_tags, dJdz_formula[sort_idx], color="black", linestyle=":",
             lw=3, label="Formula values")
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    plt.subplots_adjust(left=0.15)
    plt.grid(True)
    plt.legend(fontsize=fontsize)
    plt.xlabel("Sampled element number", fontsize=fontsize)
    plt.ylabel("Sensitivity of the objective function", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.savefig(fig_path+"Sensitivity of objective function.jpg",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Constraint function(s)
    num_consts = dgdz_formula.shape[0]
    nrows = np.floor(np.sqrt(num_consts)).astype(int)
    ncols = np.ceil(num_consts/nrows).astype(int)
    _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*8, nrows*6))
    for n in range(num_consts):
        sort_idx = np.argsort(dgdz_formula[n])
        handle = ax if num_consts == 1 else ax.flatten()[n]
        handle.plot(element_tags, dgdz_diff[n, sort_idx], color="red",
                    linestyle="-", lw=3, label="Finite difference")
        handle.plot(element_tags, dgdz_formula[n, sort_idx], color="black",
                    linestyle=":", lw=3, label="Formula values")
        handle.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        handle.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        handle.grid(True)
        handle.legend(fontsize=fontsize)
        handle.set_xlabel("Sampled element number", fontsize=fontsize)
        handle.set_ylabel(f"Sensitivity of the constraint function {n+1}",
                          fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.subplots_adjust(left=0.15)
    plt.savefig(fig_path+"Sensitivity of constraint function(s).jpg",
                dpi=300, bbox_inches="tight")
    plt.close()
