#!/usr/bin/env python3
"""High-density single-figure atlas of directional SIP contact visibility."""

from __future__ import annotations
import csv,json,math
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from _paths import RESULTS
from run_irregular_network_visibility import FULL,centres_for_branch_angle
from run_random_network_visibility_ensemble import random_network

plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
plt.rcParams['svg.fonttype']='none'
mpl.rcParams.update({'pdf.fonttype':42,'font.size':6.5,'axes.linewidth':.65,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
    'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.65,'ytick.major.width':.65})

NAVY='#1F5A85';BLUE='#3D83B5';PALE_BLUE='#DCEAF2';ORANGE='#D77835';
PALE_ORANGE='#F7E4D6';RED='#BD5149';TEAL='#2B9A8B';DARK='#2E3338';GREY='#879099';GRID='#D8DEE2'
CMAP_BLUE=LinearSegmentedColormap.from_list('sip_blue',['#F5F8FA','#AFCFE0',NAVY])
CMAP_ORANGE=LinearSegmentedColormap.from_list('sip_orange',['#FCF8F4','#EAB88E','#A94A2F'])

def panel(ax,label,x=-.06,y=1.03):
    ax.text(x,y,label,transform=ax.transAxes,fontweight='bold',fontsize=8.5,ha='left',va='bottom')

def z(r):return complex(float(r['sigma_re_S_m']),float(r['sigma_im_S_m']))

def save_all(fig,stem):
    fig.savefig(stem.with_suffix('.svg'),bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(stem.with_suffix('.tiff'),dpi=600,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(stem.with_suffix('.png'),dpi=250,bbox_inches='tight')

def draw_graph(ax,points,edges,target=None,node_color=TEAL):
    for i,j,*_ in edges:
        is_target=target is not None and {i,j}==set(target)
        ax.plot(points[[i,j],0],points[[i,j],1],color=RED if is_target else GREY,
                lw=2.3 if is_target else .8,solid_capstyle='round',zorder=1)
    ax.scatter(points[:,0],points[:,1],s=28,c=node_color,edgecolor='white',linewidth=.45,zorder=2)
    ax.set_aspect('equal');ax.axis('off')

def computation_strip(ax):
    panel(ax,'a',x=-.015,y=.94);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.02,.83,'Two-level computation',fontweight='bold',fontsize=8,va='top')
    # random-network ensemble glyphs
    for k,x0 in enumerate((.20,.285)):
        sub=ax.inset_axes([x0,.18,.085,.64]);rng=np.random.default_rng(90+k)
        pts,edges=random_network(rng,n=8);draw_graph(sub,pts,edges,node_color='#74AFC8')
    ax.text(.275,.10,'2,000 random networks',ha='center',fontsize=6.4,fontweight='bold')
    ax.annotate('',xy=(.43,.50),xytext=(.375,.50),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.2))
    ax.text(.405,.73,'screen contacts',ha='center',fontsize=5.8,color=GREY)
    ax.text(.49,.57,'33,302',ha='center',fontsize=11,fontweight='bold',color=NAVY)
    ax.text(.49,.36,'contact-direction tests',ha='center',fontsize=6.2)
    ax.annotate('',xy=(.60,.50),xytext=(.545,.50),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.2))
    # irregular FEM glyph
    sub=ax.inset_axes([.625,.15,.17,.70]);pts=np.array(centres_for_branch_angle(30))[:,:2]
    pts=(pts-pts.mean(0))*1000;draw_graph(sub,pts,FULL,target=(1,3))
    ax.text(.71,.10,'irregular 3-D FEM',ha='center',fontsize=6.4,fontweight='bold')
    ax.text(.82,.66,'4 angles × 3 frequencies',fontsize=6.2,ha='left')
    ax.text(.82,.48,'3 contact states = 36 solves',fontsize=6.2,ha='left')
    ax.text(.82,.30,'+ fine-mesh endpoint tests',fontsize=6.2,ha='left')

def computation_strip_image(ax, image_path, normalized):
    """Draw a precise two-level workflow schematic.

    The former generated banner used an unrelated dashed zoom and repeated
    particle thumbnails.  Here the upper lane is drawn from the same network
    generator as the calculations, so the selection arrow terminates on the
    network that is actually enlarged.  The lower lane then separates the
    fixed FEM geometry, the field-angle sweep, and the response output.
    """
    panel(ax,'a',x=-.015,y=.94);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    upper=ax.inset_axes([.02,.56,.96,.39]);upper.axis('off');upper.set_xlim(0,1);upper.set_ylim(0,1)
    upper.text(.02,.93,'GRAPH SCREEN',fontweight='bold',fontsize=6.4,color=NAVY,va='top')
    # Explicit ensemble: the arrow points to the enlarged network shown next.
    rng=np.random.default_rng(20260813)
    for k in range(12):
        r,c=divmod(k,4)
        sub=upper.inset_axes([.01+c*.075,.15+(2-r)*.25,.065,.19])
        pts0,edges0=random_network(rng,n=7)
        pts0=(pts0-pts0.min(0))/(np.ptp(pts0,axis=0)+1e-12);pts0=.12+.76*pts0
        sub.set_xlim(0,1);sub.set_ylim(0,1);draw_graph(sub,pts0,edges0,node_color='#74AFC8')
    upper.text(.16,.04,'2,000 random connected networks',ha='center',fontsize=5.6,fontweight='bold')
    upper.annotate('',xy=(.36,.52),xytext=(.30,.52),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.15))

    pts=np.array(centres_for_branch_angle(30))[:,:2];pts=(pts-pts.mean(0));pts/=np.max(np.abs(pts))*1.1
    selected=upper.inset_axes([.39,.10,.23,.78]);draw_graph(selected,pts,FULL,target=(1,3),node_color='#7BAFC7')
    upper.text(.505,.04,'select a target finite contact',ha='center',fontsize=5.6,fontweight='bold')
    upper.annotate('',xy=(.70,.52),xytext=(.64,.52),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.15))

    # The right-hand object is an angle sweep, not another particle ensemble.
    sweep=upper.inset_axes([.73,.10,.25,.78]);sweep.axis('off');sweep.set_xlim(0,1);sweep.set_ylim(0,1)
    sweep.text(.5,.95,'contact-direction tests',ha='center',fontsize=5.5,fontweight='bold',color=DARK)
    for j,ang in enumerate([0,30,60,90]):
        x=.14+j*.24; sweep.plot([x-.06,x+.06],[.50,.50],color=RED,lw=2.0,solid_capstyle='round')
        theta=np.deg2rad(ang);dx=.055*np.cos(theta);dy=.055*np.sin(theta)
        sweep.annotate('',xy=(x+dx,.73+dy),xytext=(x-dx,.73-dy),arrowprops=dict(arrowstyle='-|>',color=ORANGE,lw=.9))
        sweep.text(x,.22,f'{ang}°',ha='center',fontsize=4.9,color=ORANGE,fontweight='bold')
    sweep.text(.5,.04,'33,302 contact × direction cases',ha='center',fontsize=4.8,color=GREY)

    lower=ax.inset_axes([.02,.05,.96,.45]);lower.axis('off');lower.set_xlim(0,1);lower.set_ylim(0,1)
    lower.plot([0,.99],[.98,.98],color=DARK,lw=.7,clip_on=False)

    # Step 1: one fixed finite-element geometry.
    lower.text(.105,.93,'1  fixed 3-D geometry',ha='center',fontsize=5.8,fontweight='bold',color=TEAL)
    net_ax=lower.inset_axes([.015,.12,.18,.68]);draw_graph(net_ax,pts,FULL,target=(1,3),node_color='#7BAFC7')
    lower.text(.105,.06,'target contact in red',ha='center',fontsize=4.8,color=DARK)
    lower.annotate('',xy=(.235,.49),xytext=(.205,.49),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.0))

    # Step 2: four field directions applied to the same geometry.
    lower.text(.38,.93,'2  vary electric-field direction',ha='center',fontsize=5.8,fontweight='bold',color=ORANGE)
    for j,ang in enumerate([0,30,60,90]):
        x=.285+j*.065
        theta=np.deg2rad(ang);dx=.022*np.cos(theta);dy=.022*np.sin(theta)
        lower.plot([x-.025,x+.025],[.48,.48],color=RED,lw=1.5,solid_capstyle='round')
        lower.annotate('',xy=(x+dx,.74+dy),xytext=(x-dx,.74-dy),arrowprops=dict(arrowstyle='-|>',color=ORANGE,lw=.8))
        lower.text(x,.26,f'{ang}°',ha='center',fontsize=4.7,fontweight='bold',color=ORANGE)
    lower.text(.38,.11,'same geometry; only field direction changes',ha='center',fontsize=4.7,color=DARK)
    lower.annotate('',xy=(.535,.49),xytext=(.49,.49),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.0))

    # Step 3: the three contact states are explicit rather than hidden in
    # repeated particle thumbnails.
    lower.text(.625,.93,'3  repeat three contact states',ha='center',fontsize=5.8,fontweight='bold',color=DARK)
    state_ax=lower.inset_axes([.54,.14,.17,.68]);state_ax.axis('off');state_ax.set_xlim(0,1);state_ax.set_ylim(0,1)
    state_info=[('complete',RED,'-'),('target deleted',RED,':'),('disconnected',GREY,'--')]
    for y,(label,col,ls) in zip([.72,.47,.22],state_info):
        state_ax.plot([.12,.42],[y,y],color=col,lw=1.8,ls=ls,solid_capstyle='round')
        state_ax.text(.50,y,label,va='center',fontsize=4.8,color=DARK)
    lower.annotate('',xy=(.755,.49),xytext=(.715,.49),arrowprops=dict(arrowstyle='-|>',color=DARK,lw=1.0))

    # Step 4: the resulting 3 frequencies × 4 angles are compared; the
    # third contact-state dimension is represented by the preceding block.
    lower.text(.87,.93,'4  compare response',ha='center',fontsize=5.8,fontweight='bold',color=NAVY)
    resp=lower.inset_axes([.78,.12,.20,.68]);resp.imshow(normalized,aspect='auto',vmin=0,vmax=1,cmap=CMAP_BLUE)
    resp.set_xticks(range(4),['0°','30°','60°','90°'],fontsize=4.0);resp.set_yticks(range(3),['0.1','1','10'],fontsize=4.0)
    resp.tick_params(length=0,pad=1);resp.set_xlabel('field angle',fontsize=4.4,labelpad=1);resp.set_ylabel('$f/f_0$',fontsize=4.4,labelpad=1)
    resp.set_title('response contrast',fontsize=4.9,pad=2)
    for sp in resp.spines.values():sp.set_visible(False)

def main():
    root=RESULTS/'topology_research'/'random_networks'
    raw=json.loads((root/'irregular_fem_raw.json').read_text())['rows']
    ensemble=json.loads((root/'visibility_ensemble.json').read_text())
    mesh=json.loads((root/'irregular_mesh_verification.json').read_text())
    lookup={(float(r['branch_angle_deg']),float(r['frequency_over_f0']),r['case_id']):r for r in raw}
    edge=list(csv.DictReader((root/'figure_source_data'/'Fig_main_random_network_edges.csv').open()))
    single=np.array([float(r['single_fraction']) for r in edge]);orth=np.array([float(r['orthogonal_fraction']) for r in edge])
    angles=[0.,30.,60.,90.];freqs=[.1,1.,10.]
    normalized=np.zeros((3,4));share=np.zeros((3,4))
    for i,f in enumerate(freqs):
        contrast=[]
        for j,a in enumerate(angles):
            complete=z(lookup[(a,f,'complete')]);deleted=z(lookup[(a,f,'branch_deleted')]);base=z(lookup[(a,f,'disconnected')])
            c=abs(complete-deleted);contrast.append(c);share[i,j]=100*c/abs(complete-base)
        normalized[i]=np.asarray(contrast)/contrast[0]

    # The former workflow panel was schematic and repeated information stated
    # in the text and Methods.  The adopted figure is a quantitative 2 x 2
    # atlas, so each panel carries a distinct piece of the evidence chain.
    fig=plt.figure(figsize=(183/25.4,178/25.4))
    gs=fig.add_gridspec(2,2,left=.065,right=.985,bottom=.075,top=.965,
                        hspace=.36,wspace=.42)

    # a: one response matrix with a second quantitative descriptor in each cell
    holder=fig.add_subplot(gs[0,0]);panel(holder,'a');holder.axis('off')
    holder.text(0,1.025,'Selected FEM response: signal falls with angle',
                transform=holder.transAxes,fontweight='bold',fontsize=7.1,va='bottom')
    axb=holder.inset_axes([.08,.08,.76,.82]);cax=holder.inset_axes([.89,.22,.025,.54])
    im=axb.imshow(normalized,aspect='auto',vmin=0,vmax=1,cmap=CMAP_BLUE)
    axb.set_xticks(range(4),[f'{int(a)}°' for a in angles]);axb.set_yticks(range(3),[f'{f:g}' for f in freqs])
    axb.set_xlabel('tested-contact angle to electric field')
    axb.set_ylabel(r'normalized frequency, $f/f_0$')
    axb.set_title('fill: relative signal; text: share of total effect',fontsize=6.1,pad=4)
    for i in range(3):
        for j in range(4):
            val=normalized[i,j]
            text_color='white' if val>.56 else DARK
            axb.text(j,i,f'{val:.3f}\n{share[i,j]:.1f}%',ha='center',va='center',
                     fontsize=6.0,color=text_color,fontweight='bold' if j in (0,3) else 'normal',
                     linespacing=1.05)
    for sp in axb.spines.values():sp.set_visible(False)
    axb.tick_params(length=0)
    cb=fig.colorbar(im,cax=cax,ticks=[0,.25,.5,.75,1.0])
    cb.ax.tick_params(labelsize=4.8,length=1.5,pad=1)

    # b: all-edge density, hero generality panel
    axc=fig.add_subplot(gs[0,1]);panel(axc,'b')
    xp=np.maximum(single*100,1e-7);yp=np.maximum(orth*100,1e-7);lx=np.log10(xp);ly=np.log10(yp)
    hb=axc.hexbin(lx,ly,gridsize=42,bins='log',mincnt=1,cmap=CMAP_BLUE,linewidths=0)
    lo=min(lx.min(),ly.min());hi=max(lx.max(),ly.max());axc.plot([lo,hi],[lo,hi],'--',color=GREY,lw=.8)
    axc.axvline(-1,color=RED,ls=':',lw=1);axc.axhline(-1,color=NAVY,ls=':',lw=1)
    ticks=[-5,-3,-1,1];labels=[r'$10^{-5}$',r'$10^{-3}$',r'$10^{-1}$',r'$10^{1}$']
    axc.set_xticks(ticks,labels);axc.set_yticks(ticks,labels);axc.set_xlim(-5.3,1.55);axc.set_ylim(-5.3,1.55)
    axc.set(xlabel='one-direction contribution (%)',ylabel='two directions (%)',
            title='All 33,302 contacts')
    rescued=np.mean((single<.001)&(orth>=.001))*100;among=np.mean(orth[single<.001]>=.001)*100
    axc.text(-4.95,.25,f'{among:.1f}% of hidden contacts\nmove above 0.1%',color=NAVY,fontsize=6.4,
             bbox=dict(boxstyle='round,pad=.25',fc='white',ec=NAVY,lw=.7))
    axc.text(.97,.03,'colour = contact density',transform=axc.transAxes,ha='right',fontsize=5.8,color=GREY)
    cb=fig.colorbar(hb,ax=axc,fraction=.045,pad=.025);cb.set_label('contacts per hexagon',fontsize=5.8);cb.ax.tick_params(labelsize=5.5)

    # c: category transition matrix.  Use the same inset box as a so the two
    # matrices have the same displayed long/short ratio despite their
    # different numbers of categories.
    axd=fig.add_subplot(gs[1,0]);panel(axd,'c');axd.axis('off')
    axd.text(0,1.025,'Edge visibility transitions',transform=axd.transAxes,
             fontweight='normal',fontsize=7.6,va='bottom')
    cats=np.digitize(single,[.001,.01]);cato=np.digitize(orth,[.001,.01]);cm=np.zeros((3,3),int)
    for a,b in zip(cats,cato):cm[a,b]+=1
    rowpct=cm/cm.sum(1)[:,None]*100
    # Match the b-matrix's wide display box; the cell values remain a
    # row-normalized percentage, so the color scale is fixed at 0--100%.
    axdm=axd.inset_axes([.08,.05,.78,.88])
    im=axdm.imshow(rowpct,cmap=CMAP_BLUE,vmin=0,vmax=100,aspect='auto')
    names=['<0.1%\nhidden','0.1–1%\nweak','≥1%\nvisible']
    axdm.set_xticks(range(3),names);axdm.set_yticks(range(3),names)
    axdm.set(xlabel='after adding the orthogonal direction',ylabel='with one direction')
    for i in range(3):
        for j in range(3):
            axdm.text(j,i,f'{rowpct[i,j]:.1f}%\n({cm[i,j]:,})',ha='center',va='center',fontsize=6,
                     color='white' if rowpct[i,j]>48 else DARK)
    for sp in axdm.spines.values():sp.set_visible(False)
    axdm.tick_params(length=0)
    caxd=axd.inset_axes([.90,.22,.022,.52])
    cbd=fig.colorbar(im,cax=caxd,ticks=[0,25,50,75,100])
    cbd.set_label('row percentage (%)',fontsize=5.6)
    cbd.ax.tick_params(labelsize=4.8,length=1.5,pad=1)

    # d: network-level hidden-count matrix
    axe=fig.add_subplot(gs[1,1]);panel(axe,'d')
    nr=ensemble['network_rows'];ns=np.array([r['single_edges_below_0p1pct'] for r in nr]);no=np.array([r['xy_edges_below_0p1pct'] for r in nr])
    matrix=np.zeros((no.max()+1,ns.max()+1),int)
    for a,b in zip(ns,no):matrix[b,a]+=1
    ime=axe.imshow(matrix,origin='lower',aspect='auto',cmap=CMAP_ORANGE)
    axe.plot([-.5,min(ns.max(),no.max())+.5],[-.5,min(ns.max(),no.max())+.5],'--',color=GREY,lw=.8)
    axe.set(xlabel='hidden contacts: one direction',ylabel='after orthogonal direction',
            title='Hidden contacts per network')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j]>=12:axe.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=5.3,
                                        color='white' if matrix[i,j]>.55*matrix.max() else DARK)
    improved=np.mean(no<ns)*100
    axe.text(.97,.94,f'{improved:.0f}% of networks improve',transform=axe.transAxes,ha='right',va='top',
             fontsize=6.4,fontweight='bold',color=RED,bbox=dict(fc='white',ec='none',pad=1.5))
    axe.text(.97,.04,'cell colour = network count',transform=axe.transAxes,ha='right',fontsize=5.5,color=GREY)
    cbe=fig.colorbar(ime,ax=axe,fraction=.040,pad=.018)
    cbe.set_label('network count',fontsize=5.6)
    cbe.ax.tick_params(labelsize=4.8,length=1.5,pad=1)

    stem=root/'Figure_directional_blindness_atlas';save_all(fig,stem)
    submission_stem = RESULTS / 'topology_research' / 'jgr_topology_submission' / 'figures' / 'Figure_5_directional_blindness'
    save_all(fig, submission_stem);plt.close(fig)

    # Source data for derived matrices.
    out=root/'figure_source_data'
    with (out/'Fig_atlas_FEM_matrices.csv').open('w',newline='') as fh:
        w=csv.writer(fh);w.writerow(['frequency_over_f0','angle_deg','relative_signal','share_total_contact_effect_percent'])
        for i,f in enumerate(freqs):
            for j,a in enumerate(angles):w.writerow([f,a,normalized[i,j],share[i,j]])
    with (out/'Fig_atlas_transition_matrix.csv').open('w',newline='') as fh:
        w=csv.writer(fh);w.writerow(['one_direction_class','two_direction_class','count','row_percent'])
        for i in range(3):
            for j in range(3):w.writerow([names[i].replace('\n',' '),names[j].replace('\n',' '),cm[i,j],rowpct[i,j]])
    with (out/'Fig_atlas_network_hidden_counts.csv').open('w',newline='') as fh:
        w=csv.writer(fh);w.writerow(['network','hidden_one_direction','hidden_two_directions'])
        for i,(a,b) in enumerate(zip(ns,no)):w.writerow([i,a,b])
    print(json.dumps({'figure':str(stem),'rescued_all_edges_percent':rescued,
                      'rescued_among_hidden_percent':among,'networks_improved_percent':improved},indent=2))

if __name__=='__main__':main()
