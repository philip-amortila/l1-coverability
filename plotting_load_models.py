import os

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use('Agg') # matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

import pickle
import collect_baseline
import torch

# By default, the plotter saves figures to the directory where it's executed.

import base_utils
args = base_utils.get_args()

FIG_DIR = 'figs/' + args.env + '/'
model_time = args.exp_name + '/'
if not os.path.exists(FIG_DIR+model_time):
    os.makedirs(FIG_DIR+model_time)

def get_next_file(directory, model_time, ext, dot=".png"):
    i = 0
    fname = directory + model_time + ext
    while os.path.isfile(fname):
        fname = directory + model_time + ext + str(i) + dot
        i += 1
    return fname

def running_average_entropy(running_avg_entropies, running_avg_entropies_baseline):
    fname = get_next_file(FIG_DIR, model_time, "running_avg", ".png")
    plt.figure()
    plt.plot(np.arange(len(running_avg_entropies)), running_avg_entropies)
    plt.plot(np.arange(len(running_avg_entropies_baseline)), running_avg_entropies_baseline)
    plt.legend(["Cov", "Random"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Policy Entropy")
    plt.savefig(fname)

def running_average_entropy3(cov_list_of_running_avg_ent, base_list_of_running_avg_ent, ent_list_of_running_avg_ent):
    fname = get_next_file(FIG_DIR, model_time, "running_avg_ent", ".png")
    #compute statistics.
    #mean:
    cov_entropy_mean = np.mean(cov_list_of_running_avg_ent,axis=0)
    ent_entropy_mean = np.mean(ent_list_of_running_avg_ent,axis=0)
    base_entropy_mean = np.mean(base_list_of_running_avg_ent,axis=0)
    #std:
    #phil: default: ddof = 1, so \sqrt{1/(n-1) \sum_i (x_i - \bar{x})^2}
    cov_entropy_std = scipy.stats.sem(cov_list_of_running_avg_ent,ddof=0,axis=0)
    ent_entropy_std = scipy.stats.sem(ent_list_of_running_avg_ent,ddof=0,axis=0)
    base_entropy_std = scipy.stats.sem(base_list_of_running_avg_ent,ddof=0,axis=0)

    plt.figure()
    # plt.errorbar(np.arange(len(cov_entropy_mean)), cov_entropy_mean,yerr=cov_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(base_entropy_mean)), base_entropy_mean,yerr=base_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(ent_entropy_mean)), ent_entropy_mean,yerr=ent_entropy_std,fmt='-o')
    plt.plot(np.arange(len(cov_entropy_mean)), cov_entropy_mean)
    plt.plot(np.arange(len(base_entropy_mean)), base_entropy_mean)
    plt.plot(np.arange(len(ent_entropy_mean)), ent_entropy_mean)
    plt.fill_between(np.arange(len(cov_entropy_mean)), cov_entropy_mean-cov_entropy_std, cov_entropy_mean+cov_entropy_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_entropy_mean)), base_entropy_mean-base_entropy_std, base_entropy_mean+base_entropy_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_entropy_mean)), ent_entropy_mean-ent_entropy_std, ent_entropy_mean+ent_entropy_std,alpha=0.2)
    #errorbar syntax: 
        #y_error = 0.2
        #plt.plot(x, y)
        #plt.errorbar(x, y, yerr = y_error,fmt ='o')
    plt.legend(["Cov", "Random", "MaxEnt"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Policy Entropy")
    plt.savefig(fname)

def running_average_l13(cov_list_of_running_avg_l1, base_list_of_running_avg_l1, ent_list_of_running_avg_l1):
    fname = get_next_file(FIG_DIR, model_time, "running_avg_l1", ".png")
    #compute statistics.
    #mean:
    cov_l1_mean = np.mean(cov_list_of_running_avg_l1,axis=0)
    ent_l1_mean = np.mean(ent_list_of_running_avg_l1,axis=0)
    base_l1_mean = np.mean(base_list_of_running_avg_l1,axis=0)
    #std:
    #phil: default: ddof = 1, so \sqrt{1/(n-1) \sum_i (x_i - \bar{x})^2}
    cov_l1_std = scipy.stats.sem(cov_list_of_running_avg_l1,ddof=0,axis=0)
    ent_l1_std = scipy.stats.sem(ent_list_of_running_avg_l1,ddof=0,axis=0)
    base_l1_std = scipy.stats.sem(base_list_of_running_avg_l1,ddof=0,axis=0)

    plt.figure()
    # plt.errorbar(np.arange(len(cov_entropy_mean)), cov_entropy_mean,yerr=cov_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(base_entropy_mean)), base_entropy_mean,yerr=base_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(ent_entropy_mean)), ent_entropy_mean,yerr=ent_entropy_std,fmt='-o')
    plt.plot(np.arange(len(cov_l1_mean)), cov_l1_mean)
    plt.plot(np.arange(len(base_l1_mean)), base_l1_mean)
    plt.plot(np.arange(len(ent_l1_mean)), ent_l1_mean)
    plt.fill_between(np.arange(len(cov_l1_mean)), cov_l1_mean-cov_l1_std, cov_l1_mean+cov_l1_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_l1_mean)), base_l1_mean-base_l1_std, base_l1_mean+base_l1_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_l1_mean)), ent_l1_mean-ent_l1_std, ent_l1_mean+ent_l1_std,alpha=0.2)
    #errorbar syntax: 
        #y_error = 0.2
        #plt.plot(x, y)
        #plt.errorbar(x, y, yerr = y_error,fmt ='o')
    plt.legend(["Cov", "Random", "MaxEnt"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("mu-L1-coverability")
    plt.savefig(fname)

def running_average_sas3(cov_list_of_average_num_sa, base_list_of_average_num_sa, ent_list_of_average_num_sa):
    fname = get_next_file(FIG_DIR, model_time, "list_of_num_sas", ".png")
    #compute statistics.
    #mean:
    cov_sas_mean = np.mean(cov_list_of_average_num_sa,axis=0)
    ent_sas_mean = np.mean(ent_list_of_average_num_sa,axis=0)
    base_sas_mean = np.mean(base_list_of_average_num_sa,axis=0)
    #std:
    #phil: default: ddof = 1, so \sqrt{1/(n-1) \sum_i (x_i - \bar{x})^2}
    cov_sas_std = scipy.stats.sem(cov_list_of_average_num_sa,ddof=0,axis=0)
    ent_sas_std = scipy.stats.sem(ent_list_of_average_num_sa,ddof=0,axis=0)
    base_sas_std = scipy.stats.sem(base_list_of_average_num_sa,ddof=0,axis=0)

    plt.figure()
    # plt.errorbar(np.arange(len(cov_entropy_mean)), cov_entropy_mean,yerr=cov_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(base_entropy_mean)), base_entropy_mean,yerr=base_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(ent_entropy_mean)), ent_entropy_mean,yerr=ent_entropy_std,fmt='-o')
    plt.plot(np.arange(len(cov_sas_mean)), cov_sas_mean)
    plt.plot(np.arange(len(base_sas_mean)), base_sas_mean)
    plt.plot(np.arange(len(ent_sas_mean)), ent_sas_mean)
    plt.fill_between(np.arange(len(cov_sas_mean)), cov_sas_mean-cov_sas_std, cov_sas_mean+cov_sas_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_sas_mean)), base_sas_mean-base_sas_std, base_sas_mean+base_sas_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_sas_mean)), ent_sas_mean-ent_sas_std, ent_sas_mean+ent_sas_std,alpha=0.2)
    #errorbar syntax: 
        #y_error = 0.2
        #plt.plot(x, y)
        #plt.errorbar(x, y, yerr = y_error,fmt ='o')
    plt.legend(["Cov", "Random", "MaxEnt"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Number of state-actions pairs")
    plt.savefig(fname)

def running_average_pg4(cov_list_of_running_avg_pg, base_list_of_running_avg_pg, ent_list_of_running_avg_pg,none_list_of_running_avg_pg):
    fname = get_next_file(FIG_DIR, model_time, "running_avg_pg", ".png")
    #compute statistics.
    #mean:
    cov_pg_mean = np.mean(cov_list_of_running_avg_pg,axis=0)
    ent_pg_mean = np.mean(ent_list_of_running_avg_pg,axis=0)
    base_pg_mean = np.mean(base_list_of_running_avg_pg,axis=0)
    none_pg_mean = np.mean(none_list_of_running_avg_pg,axis=0)
    #std:
    #phil: default: ddof = 1, so \sqrt{1/(n-1) \sum_i (x_i - \bar{x})^2}
    cov_pg_std = scipy.stats.sem(cov_list_of_running_avg_pg,ddof=0,axis=0)
    ent_pg_std = scipy.stats.sem(ent_list_of_running_avg_pg,ddof=0,axis=0)
    base_pg_std = scipy.stats.sem(base_list_of_running_avg_pg,ddof=0,axis=0)
    none_pg_std = scipy.stats.sem(none_list_of_running_avg_pg,ddof=0,axis=0)

    plt.figure()
    # plt.errorbar(np.arange(len(cov_entropy_mean)), cov_entropy_mean,yerr=cov_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(base_entropy_mean)), base_entropy_mean,yerr=base_entropy_std,fmt='-o')
    # plt.errorbar(np.arange(len(ent_entropy_mean)), ent_entropy_mean,yerr=ent_entropy_std,fmt='-o')
    plt.plot(np.arange(len(cov_pg_mean)), cov_pg_mean)
    plt.plot(np.arange(len(base_pg_mean)), base_pg_mean)
    plt.plot(np.arange(len(ent_pg_mean)), ent_pg_mean)
    plt.plot(np.arange(len(none_pg_mean)), none_pg_mean)
    plt.fill_between(np.arange(len(cov_pg_mean)), cov_pg_mean-cov_pg_std, cov_pg_mean+cov_pg_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_pg_mean)), base_pg_mean-base_pg_std, base_pg_mean+base_pg_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_pg_mean)), ent_pg_mean-ent_pg_std, ent_pg_mean+ent_pg_std,alpha=0.2)
    plt.fill_between(np.arange(len(none_pg_mean)), none_pg_mean-none_pg_std, none_pg_mean+none_pg_std,alpha=0.2)
    #errorbar syntax: 
        #y_error = 0.2
        #plt.plot(x, y)
        #plt.errorbar(x, y, yerr = y_error,fmt ='o')
    plt.legend(["Cov", "Random", "MaxEnt","None"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Policy Opt performance")
    plt.savefig(fname)


def heatmap(running_avg_p, avg_p, i, env):
    # Create running average heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(running_avg_p))
    plt.imshow(np.ma.log(running_avg_p).filled(min_value), interpolation='spline16', cmap='Blues')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("v")
    if (env == "MountainCarContinuous-v0"):
        plt.ylabel("x")
    else:
        plt.ylabel(r"$\Theta$")
    # plt.title("Policy distribution at step %d" % i)
    running_avg_heatmap_dir = FIG_DIR + model_time + '/' + 'running_avg' + '/'
    if not os.path.exists(running_avg_heatmap_dir):
        os.makedirs(running_avg_heatmap_dir)
    fname = running_avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)

    # Create episode heatmap.
    plt.figure()
    min_value = np.min(np.ma.log(avg_p))
    plt.imshow(np.ma.log(avg_p).filled(min_value), interpolation='spline16', cmap='Blues')

    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel("v")
    if (env == "MountainCarContinuous-v0"):
        plt.ylabel("x")
    else:
        plt.ylabel(r"$\Theta$")

    # plt.title("Policy distribution at step %d" % i)
    avg_heatmap_dir = FIG_DIR + model_time + '/' + 'avg' + '/'
    if not os.path.exists(avg_heatmap_dir):
        os.makedirs(avg_heatmap_dir)
    fname = avg_heatmap_dir + "heatmap_%02d.png" % i
    plt.savefig(fname)


def heatmap4(running_avg_ps, running_avg_ps_baseline, indexes=[0,1,2,3]):
    plt.figure()
    row1 = [plt.subplot(241), plt.subplot(242), plt.subplot(243), plt.subplot(244)]
    row2 = [plt.subplot(245), plt.subplot(246), plt.subplot(247), plt.subplot(248)]

    # TODO: colorbar for the global figure
    for idx, ax in zip(indexes,row1):
        min_value = np.min(np.ma.log(running_avg_ps[idx]))
        ax.imshow(np.ma.log(running_avg_ps[idx]).filled(min_value), interpolation='spline16', cmap='Blues')
        ax.set_title("Epoch %d" % idx)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    
    for idx, ax in zip(indexes,row2):
        min_value = np.min(np.ma.log(running_avg_ps_baseline[idx]))
        ax.imshow(np.ma.log(running_avg_ps_baseline[idx]).filled(min_value), interpolation='spline16', cmap='Oranges')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    plt.tight_layout()
    fname = get_next_file(FIG_DIR, model_time, "time_heatmaps", ".png")
    plt.savefig(fname)
    # plt.colorbar()
    # plt.show()

def heatmap3x4(running_avg_ps, running_avg_ps_online, running_avg_ps_baseline, indexes=[0,1,2,3]):
    plt.figure()
    row1 = [plt.subplot(3,4,1), plt.subplot(3,4,2), plt.subplot(3,4,3), plt.subplot(3,4,4)]
    row2 = [plt.subplot(3,4,5), plt.subplot(3,4,6), plt.subplot(3,4,7), plt.subplot(3,4,8)]
    row3 = [plt.subplot(3,4,9), plt.subplot(3,4,10), plt.subplot(3,4,11), plt.subplot(3,4,12)]

    # TODO: colorbar for the global figure
    for idx, ax in zip(indexes,row1):
        min_value = np.min(np.ma.log(running_avg_ps[idx]))
        ax.imshow(np.ma.log(running_avg_ps[idx]).filled(min_value), interpolation='spline16', cmap='Blues')
        ax.set_title("Epoch %d" % idx)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    for idx, ax in zip(indexes,row2):
        min_value = np.min(np.ma.log(running_avg_ps_online[idx]))
        ax.imshow(np.ma.log(running_avg_ps_online[idx]).filled(min_value), interpolation='spline16', cmap='Greens')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
    
    for idx, ax in zip(indexes,row3):
        min_value = np.min(np.ma.log(running_avg_ps_baseline[idx]))
        ax.imshow(np.ma.log(running_avg_ps_baseline[idx]).filled(min_value), interpolation='spline16', cmap='Oranges')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    plt.tight_layout()
    fname = get_next_file(FIG_DIR, model_time, "time_heatmaps3x4", ".png")
    plt.savefig(fname)
    # plt.colorbar()
    # plt.show()

def main():

    # FIG_DIR = 
    # model_time = 
    # if not os.path.exists(FIG_DIR+model_time):
    #     os.makedirs(FIG_DIR+model_time)

    #unpickle all the data
    #aggregate them
    #call the plotting functions (which also compute the statistics)
    cov_list_of_running_avg_ent = []
    ent_list_of_running_avg_ent = []
    base_list_of_running_avg_ent = []
    cov_list_of_running_avg_l1 = []
    ent_list_of_running_avg_l1 = []
    base_list_of_running_avg_l1 = []
    cov_list_of_running_avg_pg = []
    ent_list_of_running_avg_pg = []
    base_list_of_running_avg_pg = []
    none_list_of_running_avg_pg = []
    cov_running_avg_ps = []
    ent_running_avg_ps = []
    baseline_running_avg_ps = []
    cov_list_of_average_num_sa = []
    ent_list_of_average_num_sa = []
    base_list_of_average_num_sa = []

    for i in range(args.exp_runs):
        cov_ents = []
        ent_ents = []
        base_ents = []
        cov_l1s = []
        ent_l1s = []
        base_l1s = []
        cov_pgs = []
        ent_pgs = []
        base_pgs = []
        none_pgs = []
        cov_ps = []
        ent_ps = []
        base_ps = []
        cov_sas = []
        ent_sas = []
        base_sas = []

        for j in range(args.epochs):
        #plotting.FIG_DIR + '/' + args.exp_name + '/' + args.exp_name + '_' + str(args.replicate)
            filename = FIG_DIR + '/' + args.exp_name + '/' + args.exp_name + '_' + str(i) + '_' + str(j)
            file = open(filename, 'rb')
            data = pickle.load(file)
            #append data
            #print(data)
            # for i in range(len(data)):
            #     print(data[i])
            
            cov_ents.append(data[0])
            ent_ents.append(data[1])
            base_ents.append(data[2])
            cov_l1s.append(data[3])
            ent_l1s.append(data[4])
            base_l1s.append(data[5])
            cov_pgs.append(data[6])
            ent_pgs.append(data[7])
            base_pgs.append(data[8])
            none_pgs.append(data[9])
            cov_ps.append(data[10])
            ent_ps.append(data[11])
            base_ps.append(data[12])
            #cov_ss.append(data[13])
            cov_sas.append(data[13])
            #ent_ss.append(data[13])
            ent_sas.append(data[14])
            #base_sss.append(data[15])
            base_sas.append(data[15])
            file.close()

        cov_list_of_running_avg_ent.append(cov_ents)
        ent_list_of_running_avg_ent.append(ent_ents)
        base_list_of_running_avg_ent.append(base_ents)
        cov_list_of_running_avg_l1.append(cov_l1s)
        ent_list_of_running_avg_l1.append(ent_l1s)
        base_list_of_running_avg_l1.append(base_l1s)
        cov_list_of_running_avg_pg.append(cov_pgs)
        ent_list_of_running_avg_pg.append(ent_pgs)
        base_list_of_running_avg_pg.append(base_pgs)
        none_list_of_running_avg_pg.append(none_pgs)
        cov_running_avg_ps.append(cov_ps)
        ent_running_avg_ps.append(ent_ps)
        baseline_running_avg_ps.append(base_ps)
        cov_list_of_average_num_sa.append(cov_sas)
        ent_list_of_average_num_sa.append(ent_sas)
        base_list_of_average_num_sa.append(base_sas)

    # print('cov_list_of_running_avg_l1:', cov_list_of_running_avg_l1)
    # print('ent_list_of_running_avg_l1:', ent_list_of_running_avg_l1)
    # print('base_list_of_running_avg_l1:', base_list_of_running_avg_l1)
    # print('cov_list_of_running_avg_l1[0][1::5]:', cov_list_of_running_avg_l1[0][0::5])
    # print('ent_list_of_running_avg_l1[0][1::5]:', ent_list_of_running_avg_l1[0][0::5])
    # print('base_list_of_running_avg_l1[0][1::5]:', base_list_of_running_avg_l1[0][0::5])

    # cov_list_of_running_avg_ent[1] =  cov_list_of_running_avg_ent[1][::-1]
    # ent_list_of_running_avg_ent[1] =  ent_list_of_running_avg_ent[1][::-1]
    # base_list_of_running_avg_ent[1] =  base_list_of_running_avg_ent[1][::-1]

    # print('cov_list_of_running_avg_ent:', cov_list_of_running_avg_ent)
    # print('cov_list_of_running_avg_ent.shape:', np.asarray(cov_list_of_running_avg_ent).shape)

    # print(np.asarray(cov_list_of_running_avg_ent[0]).shape)
    # print(np.asarray(cov_list_of_running_avg_ent[1][:-1]).shape)
    # print(np.asarray(cov_list_of_running_avg_l1[0][0::5]).shape)
    # print(np.asarray(cov_list_of_running_avg_l1[1][0::5][:-1]).shape)
    # print([cov_list_of_running_avg_ent[0],cov_list_of_running_avg_ent[1][:-1]])
    # l = [cov_list_of_running_avg_ent[0],cov_list_of_running_avg_ent[1][:-1]]
    # print(np.mean(l,axis=0))
    # print(cov_list_of_running_avg_l1)
    # print(ent_list_of_running_avg_l1)
    # print(base_list_of_running_avg_l1)
    print('cov_list_of_running_avg_ent:', cov_list_of_running_avg_ent)
    print('cov_list_of_running_avg_l1):', cov_list_of_running_avg_l1)
    print('cov_list_of_running_avg_pg:', cov_list_of_running_avg_pg)
    print('cov_list_of_average_num_sa:', cov_list_of_average_num_sa)
    print('cov_running_avg_ps:', cov_running_avg_ps)

    # cov_list_of_running_avg_l1 = np.asarray(cov_list_of_running_avg_l1)
    # new_cov_list = cov_list_of_running_avg_l1[:,0::5]
    # #print(new_cov_list)
    # ent_list_of_running_avg_l1 = np.asarray(ent_list_of_running_avg_l1)
    # new_ent_list = ent_list_of_running_avg_l1[:,0::5]
    # #print(new_ent_list)
    # base_list_of_running_avg_l1 = np.asarray(base_list_of_running_avg_l1)
    # new_base_list = base_list_of_running_avg_l1[:,0::5]



    #phil: load models.
    print('loading models')
    initial_state = collect_baseline.init_state(args.env)
    zero_reward = np.zeros(shape=(tuple(base_utils.num_sa)))
    LOAD_DIR = '/Users/philipamortila/Documents/GitHub/coverability-experiments-final/models-MountainCarContinuous-v0/models_mountaincar_smallepisodes_smalleps_smallinit_4_2024_01_30-00-54/'
    cov_policies = []
    ent_policies = []
    for i in range(args.epochs):
        cov_policy = torch.load(LOAD_DIR + "cov_policy"+"_"+str(i))
        ent_policy = torch.load(LOAD_DIR + "ent_policy"+"_"+str(i))
        cov_policies.append(cov_policy)
        ent_policies.append(ent_policy)

    #recompute statistics 
    cov_avg_p = []
    cov_avg_psa = []
    ent_avg_p = []
    ent_avg_psa = []
    base_avg_p = []
    base_avg_psa = []
    indices = [0,4,14,29]
    print('computing average occupancies for indices:', indices)
    #compute average occupancy
    for idx in range(args.epochs):
        cov_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        cov_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
        ent_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        ent_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
        base_eval_average_p = np.zeros(shape=(tuple(base_utils.num_states)))
        base_eval_average_psa = np.zeros(shape=(tuple(base_utils.num_sa)))
        if idx in indices:
            print('computing average occupancy for:', idx)

            for j in range(idx+1):
                cov_eval_i_p, cov_eval_i_psa,_ = cov_policies[j].execute(args.T, reward_fn=zero_reward, num_rollouts =args.num_rollouts, initial_state=initial_state, 
                    render=args.render)
                ent_eval_i_p, ent_eval_i_psa,_ = ent_policies[j].execute(args.T, reward_fn=zero_reward, num_rollouts =args.num_rollouts,initial_state=initial_state, 
                    render=args.render)
                cov_eval_average_p = cov_eval_average_p * (j)/float(j+1) + cov_eval_i_p/float(j+1)
                cov_eval_average_psa = cov_eval_average_psa * (j)/float(j+1) + cov_eval_i_psa/float(j+1)
                ent_eval_average_p = ent_eval_average_p * (j)/float(j+1) + ent_eval_i_p/float(j+1)
                ent_eval_average_psa = ent_eval_average_psa * (j)/float(j+1) + ent_eval_i_psa/float(j+1)

            base_eval_average_p, base_eval_average_psa,_  = cov_policy.execute_random(args.T, zero_reward, num_rollouts=args.num_rollouts, initial_state=initial_state, 
                render=args.render) 

        cov_avg_p.append(cov_eval_average_p)
        cov_avg_psa.append(cov_eval_average_psa)
        ent_avg_p.append(ent_eval_average_p)
        ent_avg_psa.append(ent_eval_average_psa)
        base_avg_p.append(base_eval_average_p)
        base_avg_psa.append(base_eval_average_psa)
        

    #print('cov_avg_p:', cov_avg_p)
    heatmap4(cov_avg_p, base_avg_p, indices)
    heatmap3x4(cov_avg_p, ent_avg_p, base_avg_p, indices)

    # running_average_entropy3(cov_list_of_running_avg_ent, base_list_of_running_avg_ent, ent_list_of_running_avg_ent)
    # #running_average_l13(new_cov_list, new_base_list, new_ent_list) 
    # running_average_l13(cov_list_of_running_avg_l1, base_list_of_running_avg_l1, ent_list_of_running_avg_l1) 
    # running_average_pg4(cov_list_of_running_avg_pg, base_list_of_running_avg_pg, ent_list_of_running_avg_pg,none_list_of_running_avg_pg)
    # running_average_sas3(cov_list_of_average_num_sa, base_list_of_average_num_sa, ent_list_of_average_num_sa)
    # indexes = [1,5,15,30]


    #cov_running_avg_p = cov_running_avg_ps[0][30]
    #print(len(cov_running_avg_ps))
    #print(cov_running_avg_p)
    #heatmap(cov_running_avg_p, cov_average_p, i, args.env)
    #heatmap4(cov_running_avg_ps[0], baseline_running_avg_ps[0], indexes)
    #heatmap3x4(cov_running_avg_ps[0], ent_running_avg_ps[0], baseline_running_avg_ps[0], indexes)

if __name__ == "__main__":
    main()



