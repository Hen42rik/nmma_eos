from multiprocessing import Value
import matplotlib
import matplotlib.pyplot as plt

pltparams = {"axes.grid": False,
        "text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16,
        "figure.constrained_layout.use": False}

plt.rcParams.update(pltparams)

import numpy as np
import pandas as pd

from nmma.em.likelihood import OpticalLightCurve


#############################
# DEFAULT SETTINGS / LABELS #
#############################


default_corner_kwargs = dict(bins=40, 
                        smooth=True, 
                        label_kwargs=dict(fontsize=16),
                        title_kwargs=dict(fontsize=16), 
                        quantiles=[],
                        levels=[0.68, 0.95],
                        plot_density=False, 
                        plot_datapoints=False, 
                        fill_contours=False,
                        max_n_ticks=3, 
                        min_n_ticks=3,
                        save=False,
                        truth_color="darkorange",
                        labelpad=0.2)

latex_labels=dict(inclination_EM="$\\iota$",
                  log10_E0="$\\log_{10}(E_0)$", 
                  thetaCore="$\\theta_{\\mathrm{c}}$", 
                  thetaWing="$\\theta_{\\mathrm{w}}$", 
                  alphaWing="$\\alpha_{\\mathrm{w}}$", 
                  log10_n0="$\\log_{10}(n_{\mathrm{ism}})$",
                  p="$p$", 
                  log10_epsilon_e="$\\log_{10}(\\epsilon_e)$",
                  log10_epsilon_B="$\\log_{10}(\\epsilon_B)$",
                  epsilon_e="$\\epsilon_e$",
                  epsilon_B="$\\epsilon_B$",
                  log10_mej_dyn="$\\log_{10}(m_{\\mathrm{ej,dyn}})$",
                  log10_mej_wind="$\\log_{10}(m_{\\mathrm{ej,wind}})$",
                  v_ej_dyn="$\\bar{v}_{\\mathrm{ej,dyn}}$",
                  v_ej_wind="$\\bar{v}_{\\mathrm{ej,wind}}$",
                  Ye_dyn="$\\bar{Y}_{e,\\mathrm{dyn}}$",
                  Ye_wind="$Y_{e,\\mathrm{wind}}$",
                  luminosity_distance="$d_L$",
                  redshift="$z$",
                  sys_err="$\\sigma_{\mathrm{sys}}$",
                  Gamma0="$\\Gamma_0$")


#############################
# Lightcurve plotter        #
#############################


class LightcurvePlotter:
    
    def __init__(self, 
                 posterior: dict | pd.DataFrame,
                 likelihood: OpticalLightCurve,
                 free_syserr=False,
                 ):
        
        self.systematics = "fixed"
        
        if free_syserr:
            self.systematics = "free"
        
        self.likelihood = likelihood

        self.tmin = likelihood.tmin
        self.tmax = likelihood.tmax
        self.sample_times = likelihood.sample_times
        
        self.times_det = {}
        self.mag_det = {}
        self.mag_err = {}
        
        self.times_nondet = {}
        self.mag_nondet = {}

        for key, data in likelihood.light_curve_data.items():
            mask = ~np.isinf(data[:,2])
            self.times_det[key] = data[mask,0]
            self.mag_det[key] = data[mask, 1]
            self.mag_err[key] = data[mask, 2]

            self.times_nondet[key] = data[~mask, 0]
            self.mag_nondet[key] = data[~mask, 1]

        self.model = likelihood.light_curve_model
        self.posterior = pd.DataFrame(posterior)

    def plot_data(self, 
                  ax: matplotlib.axes.Axes, 
                  filt: str,
                  zorder=3,
                  color="red",
                  **kwargs):
            
        # Detections
        t, mag, err = self.times_det[filt], self.mag_det[filt], self.mag_err[filt]
        ax.errorbar(t, mag, yerr=err, fmt="o", zorder=zorder, color=color, **kwargs)
            
        # Non-detections
        t, mag = self.times_nondet[filt], self.mag_nondet[filt]
        ax.scatter(t, mag, zorder=zorder, marker="v", color=color)


    def plot_best_fit_lc(self,
                         ax: matplotlib.axes.Axes,
                         filt: str,
                         zorder=2,
                         **kwargs):

        self._get_best_fit_lc()
        ax.plot(self.sample_times, self.best_fit_lc[filt], zorder=zorder, **kwargs)
        
    
    def _get_best_fit_lc(self,):

        if hasattr(self, "_best_fit_lc_determined"):
            return

        best_ind = np.argmax(self.posterior["log_likelihood"])
        self.best_fit_params = {}
        for key in self.posterior.keys():
            self.best_fit_params[key] = self.posterior[key][best_ind]
        
        _, self.best_fit_lc = self.model.generate_lightcurve(self.sample_times, self.best_fit_params)

        for key, value in self.best_fit_lc.items():
            self.best_fit_lc[key] = value + 5*np.log10(self.best_fit_params["luminosity_distance"]*1e6) - 5

        self._best_fit_lc_determined = True
        self.chi_squared_values()
        self.get_outliers()
    
    def plot_sys_err_band(self, 
                          ax: matplotlib.axes.Axes,
                          filt: str, 
                          zorder=2,
                          **kwargs):
        
        self._get_best_fit_lc()
        ax.fill_between(self.sample_times, 
                        self.best_fit_lc[filt] + self.best_fit_params["sys_err"],
                        self.best_fit_lc[filt] - self.best_fit_params["sys_err"],
                        alpha=0.3,
                        **kwargs)
        
    def get_outliers(self,):
        self.outliers = {}

        for filt in self.mag_det.keys():
            mag_pred = np.interp(self.times_det[filt], self.sample_times, self.best_fit_lc[filt])
            self.outliers[filt] = (mag_pred - self.mag_det[filt])**2/(self.mag_err[filt]**2+self.best_fit_params["sys_err"]**2)

    
    def chi_squared_values(self,):

        chi_squared = dict(total=0)
        reduced_chi_squared = {}
        n_data = 0
        for filt in self.mag_det.keys():

            mag_pred = np.interp(self.times_det[filt], self.sample_times, self.best_fit_lc[filt])
            chi_squared[filt] = np.sum((mag_pred - self.mag_det[filt])**2/self.mag_err[filt]**2)
            reduced_chi_squared[filt] = chi_squared[filt] / self.mag_det[filt].shape[0]

            chi_squared["total"] += chi_squared[filt]
            n_data += self.mag_det[filt].shape[0]
        
        reduced_chi_squared["total"] = chi_squared["total"] / n_data
        self.chi_squared = chi_squared
        self.reduced_chi_squared = reduced_chi_squared
    
    """
    def plot_sample_lc(self,
                       ax: matplotlib.axes.Axes,
                       filt: str,
                       zorder=1):
        
        self._get_samples_lcs()

        for j in range(200):
            ax.plot(self.t_sample_lc[j], self.sample_lc[filt][j], color="grey", alpha=0.05, zorder=zorder, rasterized=True)
    
    def _get_samples_lcs(self,):
        
        if hasattr(self, "_sample_lcs_determined"):
            return

        total_nb_samples = self.posterior.values.shape[0]
        ind = np.random.choice(total_nb_samples, 200, replace=False)

        params = {}
        for key in self.posterior.keys():
            params[key] = self.posterior[key][ind].to_numpy()
        for key in self.fixed_params:
            params[key] = np.ones(200) * self.fixed_params[key]
        
        params = self.likelihood.conversion(params)
        self.t_sample_lc, self.sample_lc = self.model.vpredict(params)
        self._sample_lcs_determined = True
    """