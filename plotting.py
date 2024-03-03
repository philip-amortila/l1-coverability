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

plt.rcParams['axes.xmargin'] = 0

import pickle

# By default, the plotter saves figures to the directory where it's executed.

import base_utils
args = base_utils.get_args()

FIG_DIR = 'figs/' + args.env + '/'
model_time = args.exp_name + '/'
if not os.path.exists(FIG_DIR+model_time):
    os.makedirs(FIG_DIR+model_time)

def get_next_file(directory, model_time, ext, dot=".pdf"):
    i = 0
    fname = directory + model_time + ext
    while os.path.isfile(fname):
        fname = directory + model_time + ext + str(i) + dot
        i += 1
    return fname

def running_average_entropy3(cov_list_of_running_avg_ent, base_list_of_running_avg_ent, ent_list_of_running_avg_ent):
    fname = get_next_file(FIG_DIR, model_time, "ent", ".pdf")
    #compute statistics.
    #mean:
    cov_entropy_mean = np.mean(cov_list_of_running_avg_ent,axis=0)
    ent_entropy_mean = np.mean(ent_list_of_running_avg_ent,axis=0)
    base_entropy_mean = np.mean(base_list_of_running_avg_ent,axis=0)
    #std:
    cov_entropy_std = scipy.stats.sem(cov_list_of_running_avg_ent,ddof=0,axis=0)
    ent_entropy_std = scipy.stats.sem(ent_list_of_running_avg_ent,ddof=0,axis=0)
    base_entropy_std = scipy.stats.sem(base_list_of_running_avg_ent,ddof=0,axis=0)

    plt.figure()
    plt.plot(np.arange(len(cov_entropy_mean)), cov_entropy_mean, linewidth=2.5)
    plt.plot(np.arange(len(base_entropy_mean)), base_entropy_mean, linewidth=2.5)
    plt.plot(np.arange(len(ent_entropy_mean)), ent_entropy_mean, linewidth=2.5)
    plt.fill_between(np.arange(len(cov_entropy_mean)), cov_entropy_mean-cov_entropy_std, cov_entropy_mean+cov_entropy_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_entropy_mean)), base_entropy_mean-base_entropy_std, base_entropy_mean+base_entropy_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_entropy_mean)), ent_entropy_mean-ent_entropy_std, ent_entropy_mean+ent_entropy_std,alpha=0.2)
    plt.title('Policy Cover Entropy', fontsize=16.5)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Entropy", fontsize=14)
    plt.xticks([0,5,10,15],fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(fname, dpi=300, bbox_inches="tight")

def running_average_l13(cov_list_of_running_avg_l1, base_list_of_running_avg_l1, ent_list_of_running_avg_l1):
    fname = get_next_file(FIG_DIR, model_time, "l1-cov", ".pdf")
    #compute statistics.
    #mean:
    cov_l1_mean = np.mean(cov_list_of_running_avg_l1,axis=0)
    ent_l1_mean = np.mean(ent_list_of_running_avg_l1,axis=0)
    base_l1_mean = np.mean(base_list_of_running_avg_l1,axis=0)
    #std error:
    cov_l1_std = scipy.stats.sem(cov_list_of_running_avg_l1,ddof=0,axis=0)
    ent_l1_std = scipy.stats.sem(ent_list_of_running_avg_l1,ddof=0,axis=0)
    base_l1_std = scipy.stats.sem(base_list_of_running_avg_l1,ddof=0,axis=0)

    plt.figure()
    plt.plot(np.arange(len(cov_l1_mean)), cov_l1_mean, linewidth=2.5)
    plt.plot(np.arange(len(base_l1_mean)), base_l1_mean, linewidth=2.5)
    plt.plot(np.arange(len(ent_l1_mean)), ent_l1_mean, linewidth=2.5)
    plt.fill_between(np.arange(len(cov_l1_mean)), cov_l1_mean-cov_l1_std, cov_l1_mean+cov_l1_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_l1_mean)), base_l1_mean-base_l1_std, base_l1_mean+base_l1_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_l1_mean)), ent_l1_mean-ent_l1_std, ent_l1_mean+ent_l1_std,alpha=0.2)
    #plt.legend([r'$L_1$-Cov', "Uniform", "MaxEnt"], prop={'size': 15})
    plt.title(r'Policy Cover $L_1$-Coverability', fontsize=16.5)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel(r'$L_1$-Coverability', fontsize=14)
    plt.xticks([0,5,10,15],fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(fname, dpi=300, bbox_inches="tight")


def running_average_ss3(cov_list_of_average_num_s, base_list_of_average_num_s, ent_list_of_average_num_s):
    fname = get_next_file(FIG_DIR, model_time, "num_states", ".pdf")
    #compute statistics.
    #mean:
    cov_ss_mean = np.mean(cov_list_of_average_num_s,axis=0)
    ent_ss_mean = np.mean(ent_list_of_average_num_s,axis=0)
    base_ss_mean = np.mean(base_list_of_average_num_s,axis=0)
    #std error:
    cov_ss_std = scipy.stats.sem(cov_list_of_average_num_s,axis=0)
    ent_ss_std = scipy.stats.sem(ent_list_of_average_num_s,axis=0)
    base_ss_std = scipy.stats.sem(base_list_of_average_num_s,axis=0)

    plt.figure()
    plt.gca().set_aspect(0.21)
    plt.plot(np.arange(len(cov_ss_mean)), cov_ss_mean, linewidth=2.5)
    plt.plot(np.arange(len(base_ss_mean)), base_ss_mean, linewidth=2.5)
    plt.plot(np.arange(len(ent_ss_mean)), ent_ss_mean, linewidth=2.5)
    plt.fill_between(np.arange(len(cov_ss_mean)), cov_ss_mean-cov_ss_std, cov_ss_mean+cov_ss_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_ss_mean)), base_ss_mean-base_ss_std, base_ss_mean+base_ss_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_ss_mean)), ent_ss_mean-ent_ss_std, ent_ss_mean+ent_ss_std,alpha=0.2)
    #plt.legend([r'$L_1$-Cov', "Uniform", "MaxEnt"], prop={'size': 15})
    plt.title('Unique States Visited', fontsize=16.5)
    plt.xlabel("Epochs", fontsize=13,labelpad=0)
    plt.ylabel("Number of states", fontsize=13,labelpad=-5)
    plt.xticks([0,5,10,15],fontsize=13)
    plt.yticks(fontsize=14)
    plt.savefig(fname, dpi=300, bbox_inches="tight")

def running_average_sas3(cov_list_of_average_num_sa, base_list_of_average_num_sa, ent_list_of_average_num_sa):
    fname = get_next_file(FIG_DIR, model_time, "num_sa_pairs", ".pdf")
    #compute statistics.
    #mean:
    cov_ss_mean = np.mean(cov_list_of_average_num_sa,axis=0)
    ent_ss_mean = np.mean(ent_list_of_average_num_sa,axis=0)
    base_ss_mean = np.mean(base_list_of_average_num_sa,axis=0)
    #std error:
    cov_ss_std = scipy.stats.sem(cov_list_of_average_num_sa,axis=0)
    ent_ss_std = scipy.stats.sem(ent_list_of_average_num_sa,axis=0)
    base_ss_std = scipy.stats.sem(base_list_of_average_num_sa,axis=0)

    plt.figure()
    plt.plot(np.arange(len(cov_ss_mean)), cov_ss_mean)
    plt.plot(np.arange(len(base_ss_mean)), base_ss_mean)
    plt.plot(np.arange(len(ent_ss_mean)), ent_ss_mean)
    plt.fill_between(np.arange(len(cov_ss_mean)), cov_ss_mean-cov_ss_std, cov_ss_mean+cov_ss_std,alpha=0.2)
    plt.fill_between(np.arange(len(base_ss_mean)), base_ss_mean-base_ss_std, base_ss_mean+base_ss_std,alpha=0.2)
    plt.fill_between(np.arange(len(ent_ss_mean)), ent_ss_mean-ent_ss_std, ent_ss_mean+ent_ss_std,alpha=0.2)
    plt.legend([r'$L_1$-Cov', "Uniform", "MaxEnt"])
    plt.xlabel("Number of Epochs")
    plt.ylabel("Unique State-actions Visisted")
    plt.savefig(fname, dpi=300)

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
    fname = running_avg_heatmap_dir + "heatmap_%02d.pdf" % i
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
    fname = avg_heatmap_dir + "heatmap_%02d.pdf" % i
    plt.savefig(fname)

def heatmap3x4(running_avg_ps, running_avg_ps_online, running_avg_ps_baseline, indexes=[0,1,2,3]):
    plt.figure(figsize=(4,4))
    row1 = [plt.subplot(3,4,1), plt.subplot(3,4,2), plt.subplot(3,4,3), plt.subplot(3,4,4)]
    row2 = [plt.subplot(3,4,5), plt.subplot(3,4,6), plt.subplot(3,4,7), plt.subplot(3,4,8)]
    row3 = [plt.subplot(3,4,9), plt.subplot(3,4,10), plt.subplot(3,4,11), plt.subplot(3,4,12)]

    # TODO: colorbar for the global figure
    for idx, ax in zip(indexes,row1):
        min_value = np.min(np.ma.log(running_avg_ps[idx]))
        ax.imshow(np.ma.log(running_avg_ps[idx]).filled(min_value), interpolation='spline16', cmap='Blues')
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
        ax.set_title("Epoch %d" % idx, fontsize = 11.5, y =-0.475)

    plt.tight_layout()
    plt.suptitle('Occupancy Heatmap', fontsize=14)
    fname = get_next_file(FIG_DIR, model_time, "time_heatmaps3x4",".pdf")
    plt.savefig(fname, dpi=400, format="pdf", bbox_inches="tight")
    # plt.colorbar()
    # plt.show()

def main():

    
    #unpickles all the data
    #aggregate it
    #call the plotting functions (which also compute the statistics)
    cov_list_of_running_avg_ent = []
    ent_list_of_running_avg_ent = []
    base_list_of_running_avg_ent = []
    cov_list_of_running_avg_l1 = []
    ent_list_of_running_avg_l1 = []
    base_list_of_running_avg_l1 = []
    cov_running_avg_ps = []
    ent_running_avg_ps = []
    baseline_running_avg_ps = []
    cov_list_of_average_num_s = []
    ent_list_of_average_num_s = []
    base_list_of_average_num_s = []
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
        cov_ps = []
        ent_ps = []
        base_ps = []
        cov_ss = []
        ent_ss = []
        base_ss = []
        cov_sas = []
        ent_sas = []
        base_sas = []

        for j in range(args.epochs):
            filename = FIG_DIR + '/' + args.exp_name + '/' + args.exp_name + '_' + str(i) + '_' + str(j)
            file = open(filename, 'rb')
            data = pickle.load(file)
            
            cov_ents.append(data[0])
            ent_ents.append(data[1])
            base_ents.append(data[2])
            cov_l1s.append(data[3])
            ent_l1s.append(data[4])
            base_l1s.append(data[5])
            cov_ps.append(data[6])
            ent_ps.append(data[7])
            base_ps.append(data[8])
            #cov_ss.append(data[13])
            cov_ss.append(data[9])
            #ent_ss.append(data[13])
            ent_ss.append(data[10])
            #base_sss.append(data[15])
            base_ss.append(data[11])
            cov_sas.append(data[12])
            #ent_ss.append(data[13])
            ent_sas.append(data[13])
            #base_sss.append(data[15])
            base_sas.append(data[14])
            file.close()

        cov_list_of_running_avg_ent.append(cov_ents)
        ent_list_of_running_avg_ent.append(ent_ents)
        base_list_of_running_avg_ent.append(base_ents)
        cov_list_of_running_avg_l1.append(cov_l1s)
        ent_list_of_running_avg_l1.append(ent_l1s)
        base_list_of_running_avg_l1.append(base_l1s)
        cov_running_avg_ps.append(cov_ps)
        ent_running_avg_ps.append(ent_ps)
        baseline_running_avg_ps.append(base_ps)
        cov_list_of_average_num_s.append(cov_ss)
        ent_list_of_average_num_s.append(ent_ss)
        base_list_of_average_num_s.append(base_ss)
        cov_list_of_average_num_sa.append(cov_sas)
        ent_list_of_average_num_sa.append(ent_sas)
        base_list_of_average_num_sa.append(base_sas)


    running_average_entropy3(cov_list_of_running_avg_ent, base_list_of_running_avg_ent, ent_list_of_running_avg_ent)
    running_average_l13(cov_list_of_running_avg_l1, base_list_of_running_avg_l1, ent_list_of_running_avg_l1) 
    running_average_ss3(cov_list_of_average_num_s, base_list_of_average_num_s, ent_list_of_average_num_s)
    indexes = [0,4,9,14]
    fav_index = 0 #pick the run that you want to visualize the occupancies for
    heatmap3x4(cov_running_avg_ps[fav_index], ent_running_avg_ps[fav_index], baseline_running_avg_ps[fav_index], indexes)


if __name__ == "__main__":
    main()



