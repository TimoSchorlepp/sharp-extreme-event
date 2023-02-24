import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import matplotlib.ticker as ticker

########################################################################
# settings for plots

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": [],
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 13,
    "font.size": 13,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "pgf.preamble":
        r"\usepackage[utf8x]{inputenc} \
        \usepackage[T1]{fontenc} \
        \usepackage{amsmath} \
        \usepackage{amssymb} "
    }
    
mpl.rcParams.update(pgf_with_latex)
mpl.rcParams['mathtext.fontset'] = 'cm'

colors   = ['steelblue', 'seagreen', 'firebrick', 'orange', 'silver', 'lightsalmon', 'navy']
insetFontSize = 10

########################################################################
# script for Fig 1

def plotSetup(eps = 0.5):
    path = 'data'
    z = np.load(path + '/target_obs.npy')
    paths = np.load(path + '/direct_sampling_paths_eps_{}.npy'.format(eps))
    inst = np.load(path + '/inst_phi.npy')
    If = np.load(path + '/inst_act.npy')
    lbda = np.load(path + '/inst_lbda.npy')
    ric_cf = np.load(path + '/ric_cf.npy')
    evals_cf = np.load(path + '/evals_cf.npy')
    sample_prob = np.load(path + '/direct_sampling_prob_eps_{}.npy'.format(eps))
    sample_std = np.load(path + '/direct_sampling_prob_std_eps_{}.npy'.format(eps))

    conf = 0.95
    c = norm.ppf((conf + 1) / 2.)
    print('eps =', eps)
    print('Probability from direct sampling:', sample_prob)
    print('Asymptotic {}% confidence interval: [{},{}]'.format( \
                100 * conf, sample_prob - c * sample_std, sample_prob + c * sample_std))
    print('number of hits for direct sampling:', len(paths[0,0]))
    print('Probability from inst + ric:', np.sqrt(eps / 2. / np.pi) * ric_cf * np.exp(-If / eps))
    print('Probability from inst + evals:', np.sqrt(eps / 2. / np.pi) * evals_cf * np.exp(-If / eps))
    meanPath = np.sum(paths, axis = 2) / len(paths[0,0])

    fig = plt.figure(figsize = (6, 5))
    ax = fig.add_subplot(1,1,1)

    x = np.linspace(-0.5, 3., 100)
    y = np.linspace(-0.5, 3., 100)
    X, Y = np.meshgrid(x, y, indexing = 'ij')
    BX,BY = -X * Y - X, X**2 - 4. * Y
    ax.streamplot(x, y, BX.T, BY.T, color = 'grey', linewidth = 0.4, density = 1.5)

    ax.scatter(0., 0., color = 'black', zorder = 20, label = r'initial position')

    targetSet = (z - x) / 2
    ax.fill_between(x, targetSet, 10 * targetSet, color = 'white', alpha = 1, zorder = 00)
    ax.fill_between(x, targetSet, 10 * targetSet, color = 'firebrick', alpha = 0.3, zorder = 00, label = 'event set')
    ax.plot(x, targetSet, color = 'firebrick', linewidth = 3, zorder = 101)
    ax.text(1.6,0.85,r'$f^{-1} \left([z, \infty) \right)$',zorder = 102, fontsize = 15)

    ax.plot(meanPath[:,0], meanPath[:,1], color = 'orange', zorder = 10, label = 'mean', linewidth = 2., path_effects=[pe.Stroke(linewidth=4, foreground= 'black'), pe.Normal()])
    ax.plot(paths[:,0,2], paths[:,1,2], color = 'orange', alpha = 0.4, label = 'sample paths', zorder = 5)
    for i in range(85,90):
        ax.plot(paths[:,0,i], paths[:,1,i], color = 'orange', alpha = 0.3, zorder = 5)
    ax.plot(inst[:,0], inst[:,1], color = 'steelblue', zorder = 10, label = r'$\left(\phi_z(t)\right)_{t \in [0,T]}$', linewidth = 2.5, linestyle = 'dashed')

    ax.set_xlim(-0.35, 2.3)
    ax.set_ylim(-0.25, 1.)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.legend(loc = 2, framealpha = 1).set_zorder(100)
    ax.set_title(r'$\varepsilon = {}$'.format(eps))

    plt.tight_layout(rect=[0, 0., 1., 1.])
    fig.savefig('paths-eps-{}.pdf'.format(eps), dpi = 800, bbox_inches = 'tight')
    plt.close()

########################################################################
# script for Fig 2

def plotEvalSpec():
    path = 'data'
    z = np.load(path + '/target_obs.npy')
    inst = np.load(path + '/inst_phi.npy')
    If = np.load(path + '/inst_act.npy')
    lbda = np.load(path + '/inst_lbda.npy')
    evals = np.load(path + '/evals.npy').real
    i = np.linspace(1, len(evals), len(evals))

    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]

    print('Fredholm det =', np.prod(1. - evals))

    fig = plt.figure(figsize = (6, 5))

    ax = fig.add_subplot(1,1,1)
    ax.loglog(i, evals, '.', label = 'positive eigenvalues', color = colors[3])
    ax.loglog(i, -evals, 'x', label = 'negative eigenvalues', color = colors[2])
    ax.grid()
    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$\left|\mu_z^{(i)}\right|$')
    ax.set_xlim(0.7, 230)
    ax.set_ylim(1e-6, 1.)
    ax.legend(framealpha = 1.)

    ax.add_patch(patches.Rectangle((0.8, 1.2e-6), 15.5, .0014, facecolor = 'white', zorder = 2, edgecolor ='black', linewidth = 0.7))
    axins = ax.inset_axes([0.1, 0.068, 0.4, 0.38])
    axins.plot(i, np.cumprod(1 - evals), '-', color = 'steelblue')
    axins.grid()
    axins.set_xlabel(r'$m$', fontsize = insetFontSize)
    axins.xaxis.set_label_coords(1.06, .05)
    axins.set_ylabel(r'$\prod_{i = 1}^m \left(1 - \mu_z^{(i)} \right)$', rotation = 0, fontsize = insetFontSize)
    axins.yaxis.set_label_coords(0.25, 1.02)
    axins.set_ylim(1., 1.2)
    axins.tick_params(axis = 'both', which = 'major', labelsize = insetFontSize)

    plt.tight_layout(rect=[0, 0., 1., 1.])
    fig.savefig('evals-conv.pdf', dpi = 800, bbox_inches = 'tight')
    plt.close()

########################################################################
# script for Fig 4

def plotCondHists(eps = 0.5):

    path = 'data'
    obs = np.load(path + '/target_obs.npy')
    inst = np.load(path + '/inst_phi.npy')
    If = np.load(path + '/inst_act.npy')
    lbda = np.load(path + '/inst_lbda.npy')
    evals = np.load(path + '/evals.npy').real
    gamma = np.load(path + '/evals_gamma.npy')
    
    dt = 1. / len(inst[:, 0])

    meanPath = np.load(path + '/ibis_mean_eps_{}.npy'.format(eps))
    samplesTimes = np.load(path + '/ibis_samples_eps_{}.npy'.format(eps))
    times = np.load(path + '/ibis_times_eps_{}.npy'.format(eps))
    idxTimes = (times / dt).astype(int)
    reweight = np.load(path + '/ibis_reweight_eps_{}.npy'.format(eps))
    fullHist = np.load(path + '/ibis_hist_eps_{}.npy'.format(eps))
    xbins = np.load(path + '/ibis_xbins_eps_{}.npy'.format(eps))
    ybins = np.load(path + '/ibis_ybins_eps_{}.npy'.format(eps))

    def getVf(x):
        return - x[0]- x[0] * x[1], -4. * x[1] + x[0]**2

    def getNormalPDF(x, epss, tt, covv):
        return 1. / (2. * np.pi * epss * np.sqrt(np.linalg.det(covv))) * np.exp(-0.5 / epss * np.sum((x - inst[tt]) * (np.linalg.inv(covv) @ (x - inst[tt]))))

    fig, axx = plt.subplots(figsize = (13, 8), nrows = 2, ncols = 3)

    for i in range(6):
        
        if i >= 1:
            
            t_idx = idxTimes[i-1]
            t = times[i-1]
            cov = np.zeros((2,2,))
            for j in range(len(evals)):
                cov = cov + np.outer(gamma[t_idx, :, j], gamma[t_idx, :, j]) / (1. - evals[j])
            
            hist, xedges, yedges = np.histogram2d(inst[t_idx, 0] + np.sqrt(eps) * samplesTimes[i-1, 0, :], inst[t_idx, 1] + np.sqrt(eps) * samplesTimes[i-1, 1, :], bins = 60, weights = reweight)
            xbinss = 0.5 * (xedges[1:] + xedges[:-1])
            ybinss = 0.5 * (yedges[1:] + yedges[:-1])
            hist = hist / np.sum(hist * (xbinss[1] - xbinss[0]) * (ybinss[1] - ybinss[0]))
            mean = [np.sum((inst[t_idx, 0] + np.sqrt(eps) * samplesTimes[i-1, 0, :]) * reweight) / np.sum(reweight), np.sum((inst[t_idx, 1] + np.sqrt(eps) * samplesTimes[i-1, 1, :]) * reweight) / np.sum(reweight)]

            NNormBins = 200
            xxbins = np.linspace(np.amin(xbinss), np.amax(xbinss), NNormBins)
            yybins = np.linspace(np.amin(ybinss), np.amax(ybinss), NNormBins)
            XX, YY = np.meshgrid(xxbins, yybins, indexing = 'ij')
            normPDF = np.zeros((NNormBins, NNormBins))
            for j in range(NNormBins):
                for k in range(NNormBins):
                    normPDF[j,k] = getNormalPDF(np.array([xxbins[j], yybins[k]]), eps, t_idx, cov)
            
            targetSet = (obs - xxbins) / 2
            VX,VY = getVf([XX, YY])
            
            ii = i // 3
            jj = i - ii * 3
            ax = axx[ii, jj]
            
            ax.text(0.4, 1.04, r'$t = {:.2f}$'.format(round(t, 2)), transform=ax.transAxes, fontsize = 12,verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=1.0))
            p1 = ax.imshow(np.ma.masked_where(hist.T == 0., hist.T), origin = 'lower', extent = (np.amin(xbinss), np.amax(xbinss), np.amin(ybinss), np.amax(ybinss)), cmap = 'Oranges')
            p2 = ax.contour(XX, YY, normPDF, colors = 'black', vmin = np.amin(hist[hist != 0.]), vmax = np.amax(hist))
            fmt = r'%.2f'
            ax.clabel(p2, p2.levels, inline=True, fmt=fmt, fontsize=10)
            ax.scatter([mean[0]], [mean[1]], marker = 'x', color = 'orange')
            ax.scatter([inst[t_idx, 0]], [inst[t_idx,1]], marker = 'x', color = 'white')
            ax.scatter([0.], [0.], color = 'black', zorder = 100)
            cb = fig.colorbar(p1, ax = ax)
            cb.ax.set_title(r'$\rho$', fontsize=12)
            cb.ax.tick_params(labelsize=10)
            ax.plot(inst[:,0], inst[:,1], color = 'white', zorder = 10)
            ax.plot(inst[:,0], inst[:,1], '--', color = 'steelblue', zorder = 10)
            ax.plot(xxbins, targetSet, color = 'firebrick', linewidth = 2)
            ax.streamplot(xxbins, yybins, VX.T, VY.T, color = (0.4,)*3, linewidth = 0.4, density = 0.8)
            if ii == 1:
                ax.set_xlabel(r'$x_1$')
            if jj == 0:
                ax.set_ylabel(r'$x_2$')
            ax.set_xlim(np.amin(xbinss), np.amax(xbinss))
            ax.set_ylim(np.amin(ybinss), np.amax(ybinss))
            ax.set_aspect('auto')
        
        if i == 0:
            
            XX, YY = np.meshgrid(xbins, ybins, indexing = 'ij')
            fullHist = fullHist / np.sum(fullHist * (xbins[1] - xbins[0]) * (ybins[1] - ybins[0]))
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                p1 = axx[0,0].imshow(np.ma.masked_where(fullHist.T == 0., np.log(fullHist.T)), origin = 'lower', extent = (np.amin(xbins), np.amax(xbins), np.amin(ybins), np.amax(ybins)), cmap = 'Oranges')
            cb = fig.colorbar(p1, ax = axx[0,0])
            cb.ax.set_title(r'$\log \rho$', fontsize=12)
            cb.ax.tick_params(labelsize=10)
            VX,VY = getVf([XX, YY])
            targetSet = (obs - xbins) / 2

            axx[0,0].streamplot(xbins, ybins, VX.T, VY.T, color = (0.4,)*3, linewidth = 0.4, density = 0.8)
            axx[0,0].scatter([0.], [0.], color = 'black', zorder = 100)
            axx[0,0].plot(inst[:,0], inst[:,1], color = 'white', zorder = 10)
            axx[0,0].plot(inst[:,0], inst[:,1], '--', color = 'steelblue', zorder = 10)
            axx[0,0].plot(meanPath[:,0], meanPath[:,1], '-', color = 'orange', zorder = 10)
            axx[0,0].plot(xbins, targetSet, color = 'firebrick', linewidth = 2)
            axx[0,0].set_aspect('auto')
            axx[0,0].set_xlim(np.amin(xbins), np.amax(xbins))
            axx[0,0].set_ylim(np.amin(ybins), np.amax(ybins))
            axx[0,0].set_ylabel(r'$x_2$')
            
            dotted_line1 = mpl.lines.Line2D([], [], linewidth=2, linestyle="--", dashes=(10, 1), color='steelblue')
            lines = [mpl.lines.Line2D([0], [0], color='black', linewidth=2, linestyle='-'), \
                     dotted_line1, \
                     mpl.lines.Line2D([0], [0], color='firebrick', linewidth=2, linestyle='-'),\
                     mpl.lines.Line2D([0], [0], color='black', linewidth=2, linestyle='None', marker = 'o'),\
                     mpl.lines.Line2D([0], [0], color='white', linewidth=2, linestyle='None', marker = 'x'),\
                     mpl.lines.Line2D([0], [0], color='orange', linewidth=2, linestyle='None', marker = 'x')]
            labels = [r'PDF levels sets for ${\cal N}\left(\phi_z(t), \varepsilon {\cal C}_z(t,t)\right)$', r'instanton trajectory $\left(\phi_z(t) \right)_{t \in [0,T]}$', r'target set $f^{-1}(\{z\})$', r'initial position $x$', r'current instanton position $\phi_z(t)$', r'current mean $X_t^\varepsilon$']
            leg = axx[0,0].legend(lines, labels, facecolor = 'gainsboro', ncol = 3, fancybox = True, loc = 'upper center', bbox_to_anchor=  (2., 1.35))
        
    plt.savefig('filter-pdfs-eps-{}.pdf'.format(eps), bbox_inches = 'tight')
    plt.close()

########################################################################

if __name__ == '__main__':
    plotSetup(eps = 0.5)
    plotEvalSpec()
    plotCondHists(eps = 0.5)
