#!/usr/bin/env python3
"""High-density atlas for the independently closed numerical study."""
from __future__ import annotations
import csv,json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from _paths import RESULTS

plt.rcParams['font.family']='sans-serif';plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans'];plt.rcParams['svg.fonttype']='none'
mpl.rcParams.update({'pdf.fonttype':42,'font.size':6.5,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
BLUE='#286D9B';TEAL='#2B978A';ORANGE='#D87832';RED='#BC5149';DARK='#30363A';GREY='#879198';LIGHT='#EEF3F5'
CMAP=LinearSegmentedColormap.from_list('supp',['#F6F9FA','#BBD7E5','#5A99BE','#194F78'])
def panel(ax,s,x=-.06,y=1.03):ax.text(x,y,s,transform=ax.transAxes,fontweight='bold',fontsize=8.5,ha='left',va='bottom')
def save(fig,stem):
 for e in ('svg','pdf'):fig.savefig(stem.with_suffix('.'+e),bbox_inches='tight')
 fig.savefig(stem.with_suffix('.tiff'),dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'});fig.savefig(stem.with_suffix('.png'),dpi=250,bbox_inches='tight')
def graph(ax,g):
 p=np.array(g['local_points_m'])*1000
 for i,j in g['core_edges']+[g['target_edge']]:
  test=[i,j]==g['target_edge'];ax.plot(p[[i,j],0],p[[i,j],1],color=RED if test else GREY,lw=2.2 if test else .8)
 ax.scatter(p[:,0],p[:,1],s=25,c=TEAL,edgecolor='white',linewidth=.4,zorder=3);ax.set_aspect('equal');ax.axis('off')
def main():
 root=RESULTS/'topology_research';ens=json.loads((root/'fem_ensemble'/'raw_results.json').read_text())
 metrics=list(csv.DictReader((root/'fem_ensemble'/'ensemble_metrics.csv').open()));analysis=json.loads((root/'fem_ensemble'/'ensemble_analysis.json').read_text())
 mesh=json.loads((root/'fem_ensemble'/'mesh_verification.json').read_text());scale=list(csv.DictReader((root/'scale_similarity'/'similarity_metrics.csv').open()))
 M=np.zeros((3,6));S=np.zeros_like(M)
 for r in metrics:
  i={.1:0,1.:1,10.:2}[float(r['frequency_over_f0'])];j=int(r['network']);M[i,j]=100*float(r['perpendicular_over_parallel']);S[i,j]=float(r['suppression_factor_parallel_over_perpendicular'])
 fig=plt.figure(figsize=(183/25.4,115/25.4));gs=fig.add_gridspec(2,15,height_ratios=[.75,1.45],left=.055,right=.985,bottom=.105,top=.97,hspace=.66,wspace=.82)
 ax=fig.add_subplot(gs[0,:]);panel(ax,'a',-.015,.96);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.text(.01,.96,'Full 3-D FEM ensemble: different irregular cores, the same tested terminal contact',fontweight='bold',fontsize=8,va='top')
 for n,g in enumerate(ens['geometries']):
  sub=ax.inset_axes([.025+n*.161,.15,.13,.67]);graph(sub,g);ax.text(.09+n*.161,.10,f'network {n+1}',ha='center',fontsize=6.2,fontweight='bold')
 ax.text(.985,.78,'6 networks × 2 directions × 3 frequencies × 2 states = 72 FEM solves',ha='right',fontsize=6.2,color=GREY)
 ax=fig.add_subplot(gs[1,:5]);panel(ax,'b');im=ax.imshow(M,aspect='auto',cmap=CMAP,norm=LogNorm(vmin=.02,vmax=6));ax.set_xticks(range(6),[f'N{i}' for i in range(1,7)]);ax.set_yticks(range(3),['0.1','1','10']);ax.set(xlabel='irregular network',ylabel=r'normalized frequency, $f/f_0$',title='Perpendicular signal as % of parallel')
 for i in range(3):
  for j in range(6):ax.text(j,i,f'{M[i,j]:.2f}%\n({S[i,j]:.0f}×)',ha='center',va='center',fontsize=5.9,color='white' if M[i,j]<.8 or M[i,j]>3 else DARK)
 for sp in ax.spines.values():sp.set_visible(False)
 cbb=fig.colorbar(im,ax=ax,fraction=.035,pad=.018,ticks=[.02,.1,1,5])
 cbb.ax.set_yticklabels(['0.02','0.1','1','5'])
 cbb.ax.set_title('%',fontsize=5.0,pad=1)
 cbb.ax.tick_params(labelsize=4.8,length=1.5,pad=1)
 ax.text(.01,-.28,'94.4% below 5%; median suppression = 75×',transform=ax.transAxes,color=RED,fontweight='bold',fontsize=6.4)
 ax=fig.add_subplot(gs[1,5:10]);panel(ax,'c');groups=[np.array([float(r['perpendicular_over_parallel'])*100 for r in metrics if float(r['frequency_over_f0'])==f]) for f in (.1,1,10.)]
 for x,q in enumerate(groups):
  ax.scatter(x+np.linspace(-.16,.16,len(q)),q,s=25,c=BLUE,edgecolor='white',linewidth=.45,zorder=3)
  ax.plot([x-.27,x+.27],[np.median(q)]*2,color=RED,lw=1.6)
  ax.plot([x,x],[q.min(),q.max()],color='#A9CADC',lw=1,zorder=1)
 ax.axhline(5,color=RED,ls='--',lw=.8);ax.set_yscale('log');ax.set_xticks([0,1,2],['0.1','1','10']);ax.set(xlabel=r'$f/f_0$',ylabel='',title='Six-network distribution\n(perpendicular / parallel, %)',ylim=(.015,7))
 ax.text(.97,.95,'red = median',transform=ax.transAxes,ha='right',va='top',fontsize=5.7,color=RED)
 ax=fig.add_subplot(gs[1,10:]);panel(ax,'d');scales=[.75,1,1.25];freqs=[.1,1,10];R=np.zeros((3,3));D=np.zeros((3,3))
 for i,s in enumerate(scales):
  for j,f in enumerate(freqs):
   a=next(float(r['delta_abs']) for r in scale if float(r['scale'])==s and float(r['angle_deg'])==0 and float(r['frequency_over_f0'])==f);b=next(float(r['delta_abs']) for r in scale if float(r['scale'])==s and float(r['angle_deg'])==90 and float(r['frequency_over_f0'])==f);ref=next(float(r['delta_abs']) for r in scale if float(r['scale'])==1 and float(r['angle_deg'])==0 and float(r['frequency_over_f0'])==f);R[i,j]=100*b/a;D[i,j]=100*abs(a-ref)/ref
 imd=ax.imshow(R,cmap=CMAP,vmin=0,vmax=1.6,aspect='auto');ax.set_xticks(range(3),['0.1','1','10']);ax.set_yticks(range(3),['0.75','1.00','1.25']);ax.set(xlabel=r'$f/f_0$',ylabel='scale, s',title='Scale similarity')
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{R[i,j]:.2f}%\nΔalign {D[i,j]:.1f}%',ha='center',va='center',fontsize=5.8,color='white' if R[i,j]>1 else DARK)
 for sp in ax.spines.values():sp.set_visible(False)
 cbd=fig.colorbar(imd,ax=ax,fraction=.050,pad=.030,ticks=[0,.4,.8,1.2,1.6])
 cbd.set_label('ratio (%)',fontsize=5.2)
 cbd.ax.tick_params(labelsize=4.5,length=1.5,pad=1)
 ax.text(.5,-.27,'aligned response differs by ≤1.36%',transform=ax.transAxes,ha='center',fontsize=5.8,color=RED,fontweight='bold')
 stem=root/'fem_ensemble'/'Figure_numerical_closure_atlas';save(fig,stem)
 submission_stem = RESULTS/'topology_research'/'jgr_topology_submission'/'figures'/'Figure_6_numerical_closure'
 save(fig,submission_stem);plt.close(fig);print(json.dumps({'figure':str(stem),'median_suppression':analysis['overall_median_suppression_factor']},indent=2))
if __name__=='__main__':main()
