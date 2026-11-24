"""Read the finished primary scientific results, apply the frozen tests, and draw the diagnostics."""
import json,platform,scipy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_figure(fig,name):
    from io import BytesIO
    from PIL import Image
    src=BytesIO();fig.savefig(src,format="png",dpi=120);src.seek(0)
    dst=BytesIO()
    Image.open(src).convert("RGB").quantize(colors=64).save(dst,format="PNG",optimize=True)
    data=dst.getvalue();assert len(data)<90*1024
    Path(name).write_bytes(data)

def run(base):
    def result(stage):return json.loads((base/stage/'results.json').read_text())
    pr=[result('probe'+str(i)) for i in range(4)]
    mo=result('moments');ext=[result('ext'+str(i)) for i in range(4)]
    pa=[result('native-path'+str(i)) for i in (1,2,4)]
    nn=result('fieldcheck');ss=result('native-smooth')
    fw=result('native-filter');best=fw[-1]['readout']
    sm=result('smooth32');bad=result('pp3')[-1]['readout']
    mh=[max(row['relative'] for row in mo if row['order']==p) for p in (6,8,10)]
    polys=result("polynomial")
    tests=dict(
        polynomial_obstructions=all((x["augmented_rank"]-x["rank"]==int(x["degree"]>=3)) and x["negative_divergence_zero"] and x["image_control"] for x in polys) and len(polys)==5,
        scalar_inverse=pr[-1]['relative_scalar']<.01 and pr[-1]['wrong_scalar_sign']>pr[-1]['scalar_error']*100,
        momentum_inverse=pr[-1]['relative_momentum']<.01 and pr[-1]['wrong_momentum_sign']>pr[-1]['momentum_error']*100,
        selected_range=pr[-1]['selected_rr']<1e-12 and pr[-1]['old_rr']>.1 and pr[-1]['inner']==0 and pr[-1]['shift_outside_collar']<1e-12,
        moment_quadrature=mh[-1]<1e-3,
        protected=nn['inner_h']<1e-12 and nn['inner_K']<1e-12,
        exterior_constraints=bool(np.all(np.array([nn['H'],nn['M']])<ext[-1]['history'][0])) and ext[-1]['min_eigenvalue']>0,
        exterior_stability=all(np.isfinite([ext[2]['dh_per_amp'],ext[2]['dK_per_amp']])),
        path=all(pa[i+1]['direct_K']<pa[i]['direct_K'] and pa[i+1]['path_h']<pa[i]['path_h'] for i in (0,1)),
        absolute_tide=bool(best['positive_tidal']<best['signal']/4 and best['flat_tidal']<best['signal']/4),
        frequency_refinement=all(fw[i+1]['readout']['positive_tidal']<fw[i]['readout']['positive_tidal'] for i in (0,1)),
        wrong_tide=best['tidal_sign_control']>10*best['positive_tidal'],
        geodesic=best['vacuum_development']['geodesic_error']<best['vacuum_development']['geodesic_signal']/4,
        scale=best['scale_test']['physical_curvature_error']<best['scale_test']['missed_K_scale_error']/4,
        smooth_class=all(ss[key]<1e-10 for key in ['root_reality_error','angular_interpolation_error','shell_TT_error']),
        smooth_roundtrip=all(r['readout']['positive_tidal']<r['readout']['signal']/4 for r in (sm,ss,pa[-1])),
        sharp_failure_detected=bad['positive_tidal']>bad['signal']/4)
    rows=dict(environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,arch=platform.machine()),tests=tests,probes=[[x['relative_scalar'],x['relative_momentum']] for x in pr],moment_errors=mh,
        constraints=[x['history'][-1] for x in ext],raw_check=nn,
        direct_pp=[[x['upper'],x['readout']['positive_tidal'],x['readout']['flat_tidal'],x['readout']['vacuum_development']['geodesic_error']] for x in fw],
        path=[[x['steps'],x['direct_h'],x['direct_K'],x['path_h'],x['path_K']] for x in pa])
    fig,axes=plt.subplots(2,2,figsize=(10,7.5),layout='constrained')
    def series(ax,x,y,lab):ax.semilogy(x,y,'o-',label=lab)
    ax=axes[0,0]
    for i,label in enumerate(['Scalar curvature','Momentum']):series(ax,[1,2,3,4],np.array(rows['probes'])[:,i],label)
    ax.axhline(.01,color='gray',ls='--',label='1% criterion');ax.set(xlabel='Mesh level',ylabel='Relative operator residual')
    ax=axes[0,1]
    for i,label in enumerate(['Hamiltonian','Momentum']):series(ax,[1,2,3],np.array(rows['constraints'])[:3,i],label)
    ax.set(xlabel='Extension mesh level',ylabel='Exterior maximum residual')
    ax=axes[1,0]
    for col,label in [(1,'Rising tidal error'),(2,'Flat-side leakage'),(3,'Geodesic error')]:
        series(ax,[48,64,96],np.array(rows['direct_pp'])[:,col],label)
    ax.axhline(best['signal']/4,color='gray',ls='--',label='Tidal criterion')
    ax.set(xlabel='Spectral window upper edge',ylabel='Absolute error')
    ax=axes[1,1]
    for col,label in [(3,'Metric'),(4,'Extrinsic curvature')]:series(ax,[1,2,4],np.array(rows['path'])[:,col],label)
    ax.set(xlabel='Midpoint steps per segment',ylabel='Difference between paths')
    for ax in axes.ravel():
        ax.grid(True,alpha=.22);ax.legend(fontsize=8)
    save_figure(fig,'diagnostics.png')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.8,3.6),layout='constrained')
    vacuum=best['vacuum_development']
    ax.plot(vacuum['profile_u'],vacuum['profile_truth'],lw=2.5,label='Input plane wave')
    ax.plot(vacuum['profile_u'],vacuum['profile_fit'],'--',label='CR reconstruction')
    ax.set(xlabel='Marked null coordinate u',ylabel='Tidal profile');ax.legend();ax.grid(True,alpha=.22)
    save_figure(fig,'waveform.png')
    return rows
