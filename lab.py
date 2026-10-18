"""Independent rank calibration, paired diagnostics, and held-out optical risk."""
from pathlib import Path
import json
import numpy as np
import optics as o
import clock_model as c
import repeat_model as r
from modes import mode_sequence,mode_repeats,BASE,SPLIT,AFFINE
from diagnostic import features,scores
from characterization import fit_rho
from threshold import rank_threshold


def blocks(rng,n,modes,factors):
    return [mode_sequence(rng.normal(size=(n,2)),modes[g],factors[g],[0,0]) for g in range(2)]


def training(n,seed):
    rng=np.random.default_rng(seed);factors=np.linalg.cholesky(c.covariances(4.)[:2])
    result={name:[] for name in ('scalar','spectrum')}
    for _ in range(n):
        s=scores(features(blocks(rng,80,BASE,factors)),features(blocks(rng,40,BASE,factors)))
        for name in result:result[name].append(s[name])
    return result


def policy_covariance(repeats,old,fresh,triggered):
    """Same budget for either alarm: 160 old+80 pilot+80 refresh+40*8=640,
    or 160 old+80 pilot+40*10=640. Pilot never enters whitening estimate.
    estimate returns class covariances: explicitly expand to the 40 event labels.
    """
    m=8 if triggered else 10;rho=fresh if triggered else old
    return r.estimate(repeats[:,:m],rho[r.LABELS])[r.LABELS]


def evaluate(trials,seed,thresholds):
    rows=[]
    for name,old_modes,current,is_null in [('matched null',BASE,BASE,True),
            ('two-mode null',SPLIT,SPLIT,True),('mode transition',BASE,SPLIT,False)]:
        # Reinitialize by scenario to pair primitive normals, not to tune outcomes.
        rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
        old_rng,pilot_rng,fresh_rng,repeat_rng,eval_rng,image_rng=rngs
        cov=c.covariances(4.);factors=np.linalg.cholesky(cov);class_factors=factors[:2]
        means=np.column_stack((c.SYNC,c.RATE*c.SYNC+c.OFFSET))
        center=o.ORIGIN+o.VELOCITY*(c.RATE*(c.FRAME+c.DELAY)+c.OFFSET)
        alarms={k:[] for k in thresholds};values={k:[] for k in thresholds}
        errors={k:[] for k in ('scalar','spectrum','never','always','oracle')}
        max_affine={k:0. for k in thresholds};centers=[];boundary=[]
        for _ in range(trials):
            old=blocks(old_rng,80,old_modes,class_factors);pilot=blocks(pilot_rng,40,current,class_factors)
            old_f,pilot_f=features(old),features(pilot);s=scores(old_f,pilot_f)
            transformed=scores(features([z@AFFINE.T+[3,5] for z in old]),
                               features([z@AFFINE.T+[3,5] for z in pilot]))
            flags={k:s[k]>thresholds[k] for k in thresholds}
            for k in thresholds:
                max_affine[k]=max(max_affine[k],abs(s[k]-transformed[k]))
                if flags[k] != (transformed[k]>thresholds[k]):raise AssertionError('Affine decision changed')
                alarms[k].append(flags[k]);values[k].append(s[k])
            fresh=np.array([fit_rho(z) for z in blocks(fresh_rng,40,current,class_factors)])
            boundary.append(np.abs(np.r_[old_f['scalar'],pilot_f['scalar'],fresh])>=.95-1e-7)
            repeats=mode_repeats(repeat_rng.normal(size=(40,10,2)),current[r.LABELS],factors,means)
            z=means+np.einsum('nij,nj->ni',factors,eval_rng.normal(size=(40,2)))
            u=o.fit_center(o.image(center)+image_rng.normal(0,o.READ_SD,len(o.PIXELS)));centers.append(u)
            # Both potential branches use independent fresh data; an untriggered
            # policy does not receive or pay for the simulated refresh control.
            fits={flag:c.calibrate(*z.T,policy_covariance(repeats,old_f['scalar'],fresh,flag)) for flag in (False,True)}
            for k in errors:
                fit=c.calibrate(*z.T,cov) if k=='oracle' else fits[flags[k] if k in flags else k=='always']
                errors[k].append(1000*(c.delay_from_center(u,fit)-c.DELAY))
        delta=np.square(errors['spectrum'])-np.square(errors['scalar'])
        rows.append(dict(scenario=name,is_null=is_null,old_modes=old_modes.tolist(),current_modes=current.tolist(),
            alarms={k:dict(rate=float(np.mean(v)),mcse=float(np.std(v,ddof=1)/np.sqrt(trials)),scores=values[k],
                           affine_max_score_difference=max_affine[k]) for k,v in alarms.items()},
            methods={k:dict(rmse_ms=float(np.sqrt(np.mean(np.square(v)))),bias_ms=float(np.mean(v))) for k,v in errors.items()},
            paired_spectrum_minus_scalar_mse_ms2=float(delta.mean()),paired_mcse_ms2=float(delta.std(ddof=1)/np.sqrt(trials)),
            errors_ms=errors,center_sd_px=float(np.std(centers,ddof=1)),rho_boundary_fraction=np.mean(boundary,axis=0).tolist()))
    return rows


def experiment(n_train=999,trials=400,seed=20261017):
    if trials<2:raise ValueError('At least two held-out trials required')
    train,test=np.random.SeedSequence(seed).spawn(2)
    values=training(n_train,train);thresholds={k:rank_threshold(v)[0] for k,v in values.items()}
    return dict(seed=seed,n_train=n_train,trials=trials,training_scores=values,thresholds=thresholds,
        rank=rank_threshold(values['scalar'])[1],total_pairs=640,
        rows=evaluate(trials,int(test.generate_state(1)[0]),thresholds),sigma_px=o.SIGMA)


def demo(out='outputs'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=experiment();out=Path(out);out.mkdir(exist_ok=True,parents=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained');x=np.arange(3);rows=result['rows']
    for k,offset in [('scalar',-.08),('spectrum',.08)]:
        ax[0].errorbar(x+offset,[v['alarms'][k]['rate'] for v in rows],yerr=[2*v['alarms'][k]['mcse'] for v in rows],fmt='o',capsize=3,label=k)
    ax[0].axhline(.05,color='gray',ls=':');ax[0].set(title='Held-out alarms',ylabel='Fraction ±2 conditional MCSE',ylim=(-.03,1.03));ax[0].legend()
    for k in ('scalar','spectrum','never','always','oracle'):
        ax[1].plot(x,[v['methods'][k]['rmse_ms'] for v in rows],'o-',label=k)
    ax[1].set(title='Optical-delay measurement',ylabel='RMSE (ms)');ax[1].legend(fontsize=8)
    ax[2].errorbar(x,[v['paired_spectrum_minus_scalar_mse_ms2'] for v in rows],yerr=[2*v['paired_mcse_ms2'] for v in rows],fmt='o',capsize=4)
    ax[2].axhline(0,color='gray',ls=':');ax[2].set(title='Spectrum minus scalar policy',ylabel='MSE difference ±2 MCSE (ms²)')
    for a in ax:a.set_xticks(x,['Matched\nnull','Two-mode\nnull','Mode\ntransition']);a.grid(axis='y',alpha=.2)
    fig.suptitle('Two lag eigenvalues: does detecting shape improve optical timing?')
    fig.savefig(out/'experiment.png',dpi=150);plt.close(fig)
    print('thresholds',result['thresholds'])
    for row in rows:print(row['scenario'],{k:v['rate'] for k,v in row['alarms'].items()},row['methods'],row['paired_spectrum_minus_scalar_mse_ms2'],row['paired_mcse_ms2'])

if __name__=='__main__':demo()
