"""
header
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../athenaGI/vis/python')
import athena_read

import pickle

gamma = 5./3.
G = 1
Mtot = 1





"""
athdf class: reading hdf5 and save what we need
"""

class athdf:
    """
    mesh are stored as attributes; e.g., self.rf, self.dS
    access variables as a dict; e.g., self['rho']
    """
    def __init__(self, fname, face_func_2=None):
        """
        face_func_2 is either a function (see MeshGen)
        or a list of thf
        """
        if isinstance(face_func_2, (list, np.ndarray)):
            f_face_func_2 = lambda a,b,c,d: face_func_2
        else:
            f_face_func_2 = face_func_2
        f = athena_read.athdf(fname, face_func_2=f_face_func_2)
        self.fname = fname
        self.face_func_2 = f_face_func_2
        self.rf, self.thf, self.phif = f['x1f'], f['x2f'], f['x3f']
        self.r, self.th, self.phi = f['x1v'], f['x2v'], f['x3v']
        self.var = {}
        for k in f.keys():
            if k[0]!='x':
                self.var[k] = f[k]
        self.init_geometry()
        return
    def __setitem__(self, key, item):
        self.var[key] = item
    def __getitem__(self, key):
        return self.var[key]
    def init_geometry(self):
        """
        save dS, dV etc.
        """
        r, th, phi, rf, thf, phif = self.r, self.th, self.phi, self.rf, self.thf, self.phif
        self.rdth = (thf[1:] - thf[:-1])[:,None] * r
        self.dS = r*r*(np.cos(thf[:-1])-np.cos(thf[1:]))[None,:,None]*(phif[1:]-phif[:-1])[:,None,None]
        self.dV = 1/3*(rf[1:]**3-rf[:-1]**3)*(np.cos(thf[:-1])-np.cos(thf[1:]))[None,:,None]*(phif[1:]-phif[:-1])[:,None,None]
        self.m0 = np.round(np.pi*2 / (phif[-1]-phif[0]))
        self.l0 = np.round(np.pi / (thf[-1]-thf[0]))
        self.OmegaK = np.sqrt(G*Mtot/self.r**3)
        return
    def int_column(self, v):
        return np.sum(v*self.rdth, axis=-2) * self.l0
    def int_dS(self, v):
        return np.sum(v*self.dS, axis=(-3,-2)) * self.l0*self.m0
    def int_dV(self, v):
        return np.sum(v*self.dV, axis=(-3,-2,-1)) * self.l0*self.m0
    def avg_phi(self, v):
        return np.mean(v, axis=-3)
    def plot_snapshot_2d(self, v, cbar=True, label='', mode='xy', bg='w', rotate_cw=0, **kwargs):
        if mode=='xy':
            xf = np.cos(self.phif-rotate_cw)[:,None] * self.rf
            yf = np.sin(self.phif-rotate_cw)[:,None] * self.rf
            plt.xlabel('x')
            plt.ylabel('y')
        elif mode=='xz':
            xf = np.sin(self.thf)[:,None] * self.rf
            yf = np.cos(self.thf)[:,None] * self.rf
            plt.xlabel('x')
            plt.ylabel('z')
        elif mode=='xz_r':
            xf = -np.sin(self.thf)[:,None] * self.rf
            yf = np.cos(self.thf)[:,None] * self.rf
            plt.xlabel('x')
            plt.ylabel('z')
        p = plt.pcolormesh(xf, yf, v, **kwargs)
        if mode[:2]=='xz' and self.l0==2:
            plt.pcolormesh(xf, -yf, v, **kwargs)
        xmax = self.rf[-1]
        plt.gca().set_facecolor(bg)
        plt.gca().set_aspect('equal')
        if cbar: plt.colorbar(label=label)
        return p
    def plot_snapshot_3d(self, v, ax=None, cbar=True, label='', slice_at_max=False, bg='w', vmin=None, vmax=None, **kwargs):
        """
        ax: 2 or 3 axes, for xy slice, xz slice, and (optionally) colorbar
            if None, make new figure
        cabr: whether to include colorbar
        label: colorbar label
        slice_at_max:
            True: phi slice at max of v;
            False: phi slice at phi=0;
            array with same shape as v: phi slice at max of slice_at_max
        """ 
        if ax is None:
            fig, ax = plt.subplots(1,3,figsize=(13,5),width_ratios=[1, 1, 0.1])
        if vmin is None: vmin = np.amin(v)
        if vmax is None: vmax = np.amax(v)
        try:
            if slice_at_max==False:
                i_phi = 0
            elif slice_at_max==True:
                i_phi = np.argmax(v)//(len(self.r)*len(self.th))
            else:
                i_phi = np.argmax(slice_at_max)//(len(self.r)*len(self.th))
        except:
            i_phi = np.argmax(slice_at_max)//(len(self.r)*len(self.th))
        if self.l0==1:
            i_mid = (len(self.th)-1)//2
        else:
            i_mid = -1
        plt.sca(ax[0])
        p = self.plot_snapshot_2d(v[:,i_mid,:], cbar=False, mode='xy', bg=bg, vmin=vmin, vmax=vmax, **kwargs)
        plt.sca(ax[1])
        self.plot_snapshot_2d(v[i_phi,:,:], cbar=False, mode='xz', bg=bg, vmin=vmin, vmax=vmax, **kwargs)
        self.plot_snapshot_2d(v[(i_phi+len(self.phi)//2)%(len(self.phi)),:,:], cbar=False, mode='xz_r', bg=bg, vmin=vmin, vmax=vmax, **kwargs)
        if cbar:
            plt.gcf().colorbar(p, cax=ax[2], label=label)
        return p, ax