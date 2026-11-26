"""Expanded cross-station DO regression comparison with source-only chronological tuning."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from forecast_core import (HORIZONS, CONFIGS, load_station, apply_do_policy,
    make_features, split_masks, target_at, build_model, scores, detection, sha, write_json)

PRIMARY_MODELS = ('linear','ridge','polynomial','decision_tree','random_forest','hist_gbm','mlp')
SECONDARY_MODELS = ('ridge','hist_gbm')
POLICIES = ('exclude_negative','clip_zero')
INPUTS = ('combined','do_history')
CUTOFF = pd.Timestamp('2024-08-07 19:00')
FOLDS = [(pd.Timestamp(a),pd.Timestamp(b)) for a,b in (
    ('2024-06-13 14:15','2024-07-02'),('2024-07-02','2024-07-20 09:30'),
    ('2024-07-20 09:30','2024-08-07 19:00'))]

HYPOXIA = 2.0          # mg/L threshold


def models_for(mode):
    return PRIMARY_MODELS if mode=='combined' else SECONDARY_MODELS


def train_model(data, features, h, name, params, mask):
    """Training data and preprocessing always belong to the same single station."""
    y,_=target_at(data,h)
    model=build_model(name,params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start=time.monotonic()
        model.fit(features.loc[mask],(y-data.do).loc[mask])
    elapsed=time.monotonic()-start
    np.testing.assert_allclose(model[0].statistics_,features.loc[mask].median().to_numpy(),atol=1e-12,rtol=0)
    estimator=model[-1].regressor_ if name=='mlp' else model[-1]
    iterations=getattr(estimator,'n_iter_',None)
    diagnostic={'fit_seconds':elapsed,'train_n':int(mask.sum()),
        'n_iter':int(iterations) if iterations is not None else None,
        'convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in caught),
        'warnings':' || '.join(str(w.message) for w in caught),
        'final_training_loss':float(estimator.loss_) if hasattr(estimator,'loss_') else None}
    return model,diagnostic


def predict_frozen(model,data,features,mask):
    medians=model[0].statistics_.copy()
    answer=data.do.loc[mask].to_numpy()+model.predict(features.loc[mask])
    np.testing.assert_array_equal(medians,model[0].statistics_)
    if not np.isfinite(answer).all():raise ValueError('Nonfinite predictions')
    return answer


def paired_interval(frame,reference,candidate,days,reps=2000):
    loss=np.abs(frame.actual-frame[candidate])-np.abs(frame.actual-frame[reference])
    blocks=(pd.DatetimeIndex(frame.origin_time)-CUTOFF)//pd.Timedelta(days=days)
    agg=pd.DataFrame({'loss':loss.to_numpy(),'block':blocks}).groupby('block').loss.agg(['sum','count'])
    draw=np.random.default_rng(53).integers(0,len(agg),size=(reps,len(agg)))
    boot=agg['sum'].to_numpy()[draw].sum(1)/agg['count'].to_numpy()[draw].sum(1)
    return {'block_days':days,'nonempty_blocks':len(agg),'resamples':reps,
        'mae_difference':float(loss.mean()),'ci_low':float(np.quantile(boot,.025)),
        'ci_high':float(np.quantile(boot,.975))}


def run(package):
    out=package/'results';out.mkdir(exist_ok=True)
    paths={s:package/'data'/f'{s}.csv' for s in ('AEB','MTI')}
    protocol={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'question':'Does a model trained at one monitoring station work at a station in a different tributary, or does each station need its own model?',
        'main_policy':'exclude_negative','sensitivity_policy':'clip_zero',
        'policy_rationale':'Project choice informed by DNR GNV masking guidance, not proof that individual CSV readings have a GNV flag. Clipping is retained for sensitivity and continuity with v1. Each policy is tuned independently using source validation only.',
        'quality_reference':'https://eyesonthebay.dnr.maryland.gov/eyesonthebay/documents/SWM_QAPP_July2023.pdf#page=92',
        'primary_input_set':'combined','secondary_input_set':'do_history',
        'models_by_input':{'combined':PRIMARY_MODELS,'do_history':SECONDARY_MODELS},
        'horizons_hours':HORIZONS,'evaluation_start':str(CUTOFF),
        'folds':[{'start':str(a),'exclusive_target_end':str(b)} for a,b in FOLDS],
        'parameter_candidates':CONFIGS,
        'model_definitions':{
            'linear':'Ordinary least squares on 24 combined features, with training-only median imputation and scaling.',
            'ridge':'Linear Ridge, alpha 1 or 100, training-only imputation and scaling.',
            'polynomial':'Degree-two terms and pairwise interactions, 324 columns from 24 inputs, with training-only scaling and Ridge alpha 1 or 100. This is regularized quadratic regression, not unregularized polynomial OLS.',
            'decision_tree':'DecisionTreeRegressor, min_samples_leaf 20, depth 5 or 10, random_state 0.',
            'random_forest':'100 trees, min_samples_leaf 20, max_features 0.8, depth 8 or unrestricted, random_state 0.',
            'hist_gbm':'150 boosting iterations, learning rate 0.05, L2 1, 7 or 15 leaves, random_state 0; internal early stopping disabled.',
            'mlp':'Feedforward neural network with two hidden ReLU layers: (32,16) or (64,32). Adam, alpha .01, batch 256, learning rate .001, max_iter 400, tol 1e-4, n_iter_no_change 25, seed 0. Inputs and delta targets standardized on source training only. No internal random validation or early stopping. Convergence warnings are retained.'},
        'selection':'Pooled MAE across three expanding chronological source-station validation folds. Select configuration within each family, then select the ML family with lowest source validation MAE before evaluation. Persistence remains a separate baseline, not a candidate ML family. No receiving data enter transferred fitting/tuning.',
        'target':'Future DO at exactly 1,3,6,12,24 elapsed hours. All regressors learn future-minus-current DO; receiving current DO is added back. Primary predictions are not clipped.',
        'features':'Combined: 9 current/history DO columns plus 15 current/lagged environmental columns (temperature, salinity, pH, turbidity, chlorophyll). No DOSAT, no station IDs, no future inputs, no exact difference redundancies. Secondary: 9 DO-history columns.',
        'matching':'Identical eligible receiving origins/targets for all forecasts within a policy, station, horizon, and input set. Missing DO origins/targets excluded, not imputed. No row deletion before lagging. Policies additionally compared on their exact intersection of nonnegative current/future receiving DO.',
        'scope':'Requires receiving current/historical sensor inputs. It is transfer without local fitting, not forecasting at an unmonitored station. Training samples differ in number between stations.',
        'metrics':'MAE primary, RMSE and R2 secondary. R2 is null for constant targets or fewer than two observations. No operational acceptance tolerance or noninferiority margin invented.',
        'uncertainty':'Exploratory paired calendar-block bootstrap, 2000 resamples, 1/2/4/7-day blocks; no multiplicity adjustment. Coverage and block dependence assumptions not established.',
        'evaluation_status':'Retrospective exploratory follow-up. Both stations\' evaluation data were viewed in v1. New choices recorded before new scoring but not a preregistration or fresh confirmatory test.',
        'data_sha256':{s:sha(p) for s,p in paths.items()},
        'code_sha256':{n:sha(package/n) for n in ('forecast_core.py','model_comparison.py')},
        'software':{x:importlib.metadata.version(x) for x in ('numpy','pandas','scipy','scikit-learn','matplotlib')},
        'limitations':['Two stations in one season; cannot establish that every station needs a local model.',
            'Coarse candidate sets, one neural-network random seed, and no sequential LSTM/Transformer models.',
            'No per-measurement quality flags or operational telemetry latency model.',
            'Strong predictor collinearity can destabilize coefficient interpretation and extrapolation.']}
    write_json(out/'protocol.json',protocol)
    raw={};audit={}
    for s,p in paths.items():
        raw[s],audit[s]=load_station(p)
        assert raw[s].index.min()==pd.Timestamp('2024-04-01')
        assert raw[s].index.max()==pd.Timestamp('2024-10-01 23:45')
    write_json(out/'data_audit.json',audit)
    datasets={(policy,s):apply_do_policy(d,policy) for policy in POLICIES for s,d in raw.items()}
    features={(policy,s,mode):make_features(d,mode) for (policy,s),d in datasets.items() for mode in INPUTS}
    fitted={};selected=[];cv=[];splits=[];family_selection=[];fit_count=0
    log_fields=['policy','source','horizon_hours','input_set','model','parameters','stage','fold',
        'fit_seconds','train_n','n_iter','convergence_warning','warnings','final_training_loss']
    with (out/'fit_diagnostics.csv').open('w',newline='') as log:
        writer=csv.DictWriter(log,fieldnames=log_fields);writer.writeheader()
        for (policy,source),d in datasets.items():
            for h in HORIZONS:
                _,_,train,_=split_masks(d,h,CUTOFF)
                for stage,start,end in [('evaluation_fit',CUTOFF,None)]+[(f'validation_{i}',a,b) for i,(a,b) in enumerate(FOLDS,1)]:
                    y,t,tr,va=split_masks(d,h,start,end)
                    splits.append({'policy':policy,'source':source,'horizon_hours':h,'stage':stage,
                        'train_n':int(tr.sum()),'evaluation_n':int(va.sum()),
                        'train_target_max':str(t[tr].max()),'evaluation_origin_min':str(d.index[va].min()),
                        'evaluation_target_max':str(t[va].max()),'exclusive_target_end':str(end) if end else ''})
                for mode in INPUTS:
                    x=features[policy,source,mode];family_scores=[]
                    for name in models_for(mode):
                        candidate_scores=[]
                        context={'policy':policy,'source':source,'horizon_hours':h,'input_set':mode,'model':name}
                        for params in CONFIGS[name]:
                            total_loss=0.;total_n=0
                            for fold,(start,end) in enumerate(FOLDS,1):
                                y,t,tr,va=split_masks(d,h,start,end)
                                model,diagnostic=train_model(d,x,h,name,params,tr)
                                writer.writerow({**context,'parameters':json.dumps(params,sort_keys=True),
                                    'stage':'validation','fold':fold,**diagnostic});log.flush();fit_count+=1
                                pred=predict_frozen(model,d,x,va);sc=scores(y.loc[va],pred)
                                cv.append({**context,'parameters':json.dumps(params,sort_keys=True),
                                    'fold':fold,**sc})
                                total_loss+=sc['mae']*sc['n'];total_n+=sc['n']
                            candidate_scores.append((total_loss/total_n,params))
                        best_mae,best=min(candidate_scores,key=lambda item:item[0])
                        model,diagnostic=train_model(d,x,h,name,best,train)
                        writer.writerow({**context,'parameters':json.dumps(best,sort_keys=True),
                            'stage':'evaluation_fit','fold':'',**diagnostic});log.flush();fit_count+=1
                        fitted[policy,source,h,mode,name]=model
                        selected.append({**context,'parameters':json.dumps(best,sort_keys=True),
                            'pooled_validation_mae':best_mae,'train_n':int(train.sum()),
                            'fit_convergence_warning':diagnostic['convergence_warning']})
                        family_scores.append((best_mae,name))
                        write_json(out/'training_progress.json',{'phase':'source-only fitting',
                            'completed_fits':fit_count,'last':context})
                    winner_mae,winner=min(family_scores,key=lambda item:item[0])
                    baseline_loss=0.;baseline_n=0
                    for fold,(start,end) in enumerate(FOLDS,1):
                        y,t,tr,va=split_masks(d,h,start,end)
                        sc=scores(y.loc[va],d.do.loc[va])
                        cv.append({'policy':policy,'source':source,'horizon_hours':h,
                            'input_set':mode,'model':'persistence','parameters':'{}','fold':fold,**sc})
                        baseline_loss+=sc['mae']*sc['n'];baseline_n+=sc['n']
                    family_selection.append({'policy':policy,'source':source,'horizon_hours':h,
                        'input_set':mode,'selected_model':winner,'pooled_validation_mae':winner_mae,
                        'persistence_validation_mae':baseline_loss/baseline_n})
                print(f'Fitted {policy}, {source}, {h}h ({fit_count} cumulative fits)',flush=True)
    pd.DataFrame(selected).to_csv(out/'selected_parameters.csv',index=False)
    pd.DataFrame(cv).to_csv(out/'validation_scores.csv',index=False)
    pd.DataFrame(splits).to_csv(out/'split_audit.csv',index=False)
    pd.DataFrame(family_selection).to_csv(out/'selected_families.csv',index=False)
    write_json(out/'selection_freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'selected_parameters_sha256':sha(out/'selected_parameters.csv'),
        'selected_families_sha256':sha(out/'selected_families.csv'),
        'note':'All source-only configuration/family selections and final model fits completed before evaluation scoring.'})
    winners={(r['policy'],r['source'],r['horizon_hours'],r['input_set']):r['selected_model'] for r in family_selection}
    predictions=[];metrics=[];intervals=[]
    for (policy,receiver),d in datasets.items():
        source=next(s for s in raw if s!=receiver)
        for h in HORIZONS:
            y,t,_,test=split_masks(d,h,CUTOFF)
            for mode in INPUTS:
                ctx={'policy':policy,'receiver':receiver,'source':source,'horizon_hours':h,'input_set':mode}
                p=pd.DataFrame({**ctx,'origin_time':d.index[test],'target_time':t[test],
                    'actual':y.loc[test].to_numpy(),'current_do':d.do.loc[test].to_numpy(),
                    'persistence':d.do.loc[test].to_numpy()})
                x=features[policy,receiver,mode]
                columns=['persistence']
                for name in models_for(mode):
                    for scope,train_station in [('local',receiver),('transferred',source)]:
                        col=f'{scope}_{name}';columns.append(col)
                        p[col]=predict_frozen(fitted[policy,train_station,h,mode,name],d,x,test)
                for scope,train_station in [('local',receiver),('transferred',source)]:
                    winner=winners[policy,train_station,h,mode]
                    p[f'{scope}_selected']=p[f'{scope}_{winner}']
                    p[f'{scope}_selected_family']=winner
                    columns.append(f'{scope}_selected')
                baseline=scores(p.actual,p.persistence)
                for col in columns:
                    sc=scores(p.actual,p[col])
                    metrics.append({**ctx,'forecast':col,**sc,
                        'mae_skill_vs_persistence':1-sc['mae']/baseline['mae'],
                        'predicted_below_zero_fraction':float((p[col]<0).mean()),
                        'prediction_min':float(p[col].min()),'prediction_max':float(p[col].max())})
                for name in [*models_for(mode),'selected']:
                    for reference in [f'local_{name}','persistence']:
                        for days in (1,2,4,7):
                            intervals.append({**ctx,'candidate':f'transferred_{name}',
                                'reference':reference,**paired_interval(p,reference,f'transferred_{name}',days)})
                predictions.append(p)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'evaluation_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_mae_differences.csv',index=False)
    # Compare independently tuned policies on exactly the same receiving timestamps.
    common=[]
    for receiver in raw:
        for h in HORIZONS:
            for mode in INPUTS:
                frames={p.policy.iloc[0]:p for p in predictions if p.receiver.iloc[0]==receiver and
                    int(p.horizon_hours.iloc[0])==h and p.input_set.iloc[0]==mode}
                left=frames['exclude_negative'].set_index('origin_time')
                right=frames['clip_zero'].set_index('origin_time').loc[left.index]
                np.testing.assert_allclose(left.actual,right.actual)
                for col in ['persistence']+[f'{scope}_{name}' for name in [*models_for(mode),'selected'] for scope in ('local','transferred')]:
                    common.append({'receiver':receiver,'horizon_hours':h,'input_set':mode,'forecast':col,
                        'common_n':len(left),'mae_exclude_negative':scores(left.actual,left[col])['mae'],
                        'mae_clip_zero_on_common':scores(right.actual,right[col])['mae']})
    pd.DataFrame(common).to_csv(out/'cleaning_policy_comparison.csv',index=False)
    write_json(out/'training_progress.json',{'phase':'complete','completed_fits':fit_count,
        'metric_rows':len(metrics),'prediction_rows':sum(len(p) for p in predictions)})
    print(f'Completed {fit_count} fits; {len(metrics)} metric rows.',flush=True)


def train_and_evaluate(package, station):
    """Train one station and return per-horizon train/test regression and event metrics."""
    out=package/'results'/'single_station'/station
    out.mkdir(parents=True,exist_ok=True)
    path=package/'data'/f'{station}.csv'
    protocol={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'question':'How does local forecast performance vary with horizon at one station?',
        'main_policy':'exclude_negative','sensitivity_policy':'clip_zero',
        'policy_rationale':'Project choice informed by DNR GNV masking guidance, not proof that individual CSV readings have a GNV flag. Clipping is retained for sensitivity and continuity with v1. Each policy is tuned independently using source validation only.',
        'quality_reference':'https://eyesonthebay.dnr.maryland.gov/eyesonthebay/documents/SWM_QAPP_July2023.pdf#page=92',
        'primary_input_set':'combined','secondary_input_set':'do_history',
        'models_by_input':{'combined':PRIMARY_MODELS,'do_history':SECONDARY_MODELS},
        'horizons_hours':HORIZONS,'evaluation_start':str(CUTOFF),
        'folds':[{'start':str(a),'exclusive_target_end':str(b)} for a,b in FOLDS],
        'parameter_candidates':CONFIGS,
        'model_definitions':{
            'linear':'Ordinary least squares on 24 combined features, with training-only median imputation and scaling.',
            'ridge':'Linear Ridge, alpha 1 or 100, training-only imputation and scaling.',
            'polynomial':'Degree-two terms and pairwise interactions, 324 columns from 24 inputs, with training-only scaling and Ridge alpha 1 or 100. This is regularized quadratic regression, not unregularized polynomial OLS.',
            'decision_tree':'DecisionTreeRegressor, min_samples_leaf 20, depth 5 or 10, random_state 0.',
            'random_forest':'100 trees, min_samples_leaf 20, max_features 0.8, depth 8 or unrestricted, random_state 0.',
            'hist_gbm':'150 boosting iterations, learning rate 0.05, L2 1, 7 or 15 leaves, random_state 0; internal early stopping disabled.',
            'mlp':'Feedforward neural network with two hidden ReLU layers: (32,16) or (64,32). Adam, alpha .01, batch 256, learning rate .001, max_iter 400, tol 1e-4, n_iter_no_change 25, seed 0. Inputs and delta targets standardized on source training only. No internal random validation or early stopping. Convergence warnings are retained.'},
        'selection':'Pooled MAE across three expanding chronological validation folds. Select configuration within each family, then select the ML family with lowest validation MAE before evaluation. Persistence remains a separate baseline.',
        'target':'Future DO at exactly 1,3,6,12,24 elapsed hours. All regressors learn future-minus-current DO; receiving current DO is added back. Primary predictions are not clipped.',
        'features':'Combined: 9 current/history DO columns plus 15 current/lagged environmental columns (temperature, salinity, pH, turbidity, chlorophyll). No DOSAT, no station IDs, no future inputs, no exact difference redundancies. Secondary: 9 DO-history columns.',
        'matching':'Identical eligible origins/targets for all forecasts within a policy, horizon, and input set. Missing DO origins/targets excluded, not imputed. No row deletion before lagging.',
        'scope':'Single-station training and evaluation. Train metrics are in-sample; test metrics use held-out future observations.',
        'metrics':'MAE, RMSE, R2, and hypoxia-event precision, recall and F1. Hypoxia is DO strictly below the threshold; undefined metrics are null.',
        'hypoxia_threshold_mg_l':HYPOXIA,
        'evaluation_status':'Retrospective exploratory analysis, not a fresh confirmatory test.',
        'data_sha256':{station:sha(path)},
        'code_sha256':{n:sha(package/n) for n in ('forecast_core.py','model_comparison.py')},
        'software':{x:importlib.metadata.version(x) for x in ('numpy','pandas','scipy','scikit-learn','matplotlib')},
        'limitations':['One station in one season; results may not generalize to other stations.',
            'Coarse candidate sets, one neural-network random seed, and no sequential LSTM/Transformer models.',
            'No per-measurement quality flags or operational telemetry latency model.',
            'Strong predictor collinearity can destabilize coefficient interpretation and extrapolation.']}

    write_json(out/'protocol.json',protocol)
    raw,audit=load_station(path)
    assert raw.index.min()==pd.Timestamp('2024-04-01')
    assert raw.index.max()==pd.Timestamp('2024-10-01 23:45')
    write_json(out/'data_audit.json',{station:audit})

    datasets={(policy,station):apply_do_policy(raw,policy) for policy in POLICIES}
    features={(policy,s,mode):make_features(d,mode) for (policy,s),d in datasets.items() for mode in INPUTS}
    fitted={};selected=[];cv=[];splits=[];family_selection=[];fit_count=0

    log_fields=['policy','source','horizon_hours','input_set','model','parameters','stage','fold',
        'fit_seconds','train_n','n_iter','convergence_warning','warnings','final_training_loss']

    with (out/'fit_diagnostics.csv').open('w',newline='') as log:
        writer=csv.DictWriter(log,fieldnames=log_fields);writer.writeheader()
        for (policy,source),d in datasets.items():
            for h in HORIZONS:
                _,_,train,_=split_masks(d,h,CUTOFF)
                for stage,start,end in [('evaluation_fit',CUTOFF,None)]+[(f'validation_{i}',a,b) for i,(a,b) in enumerate(FOLDS,1)]:
                    y,t,tr,va=split_masks(d,h,start,end)
                    splits.append({'policy':policy,'source':source,'horizon_hours':h,'stage':stage,
                        'train_n':int(tr.sum()),'evaluation_n':int(va.sum()),
                        'train_target_max':str(t[tr].max()),'evaluation_origin_min':str(d.index[va].min()),
                        'evaluation_target_max':str(t[va].max()),'exclusive_target_end':str(end) if end else ''})
                for mode in INPUTS:
                    x=features[policy,source,mode];family_scores=[]
                    for name in models_for(mode):
                        candidate_scores=[]
                        context={'policy':policy,'source':source,'horizon_hours':h,'input_set':mode,'model':name}
                        for params in CONFIGS[name]:
                            total_loss=0.;total_n=0
                            for fold,(start,end) in enumerate(FOLDS,1):
                                y,t,tr,va=split_masks(d,h,start,end)
                                model,diagnostic=train_model(d,x,h,name,params,tr)
                                writer.writerow({**context,'parameters':json.dumps(params,sort_keys=True),
                                    'stage':'validation','fold':fold,**diagnostic});log.flush();fit_count+=1
                                pred=predict_frozen(model,d,x,va);sc=scores(y.loc[va],pred)
                                cv.append({**context,'parameters':json.dumps(params,sort_keys=True),
                                    'fold':fold,**sc})
                                total_loss+=sc['mae']*sc['n'];total_n+=sc['n']
                            candidate_scores.append((total_loss/total_n,params))
                        best_mae,best=min(candidate_scores,key=lambda item:item[0])
                        model,diagnostic=train_model(d,x,h,name,best,train)
                        writer.writerow({**context,'parameters':json.dumps(best,sort_keys=True),
                            'stage':'evaluation_fit','fold':'',**diagnostic});log.flush();fit_count+=1
                        fitted[policy,source,h,mode,name]=model
                        selected.append({**context,'parameters':json.dumps(best,sort_keys=True),
                            'pooled_validation_mae':best_mae,'train_n':int(train.sum()),
                            'fit_convergence_warning':diagnostic['convergence_warning']})
                        family_scores.append((best_mae,name))
                        write_json(out/'training_progress.json',{'phase':'source-only fitting',
                            'completed_fits':fit_count,'last':context})
                    winner_mae,winner=min(family_scores,key=lambda item:item[0])
                    baseline_loss=0.;baseline_n=0
                    for fold,(start,end) in enumerate(FOLDS,1):
                        y,t,tr,va=split_masks(d,h,start,end)
                        sc=scores(y.loc[va],d.do.loc[va])
                        cv.append({'policy':policy,'source':source,'horizon_hours':h,
                            'input_set':mode,'model':'persistence','parameters':'{}','fold':fold,**sc})
                        baseline_loss+=sc['mae']*sc['n'];baseline_n+=sc['n']
                    family_selection.append({'policy':policy,'source':source,'horizon_hours':h,
                        'input_set':mode,'selected_model':winner,'pooled_validation_mae':winner_mae,
                        'persistence_validation_mae':baseline_loss/baseline_n})
                print(f'Fitted {policy}, {source}, {h}h ({fit_count} cumulative fits)',flush=True)

    pd.DataFrame(selected).to_csv(out/'selected_parameters.csv',index=False)
    pd.DataFrame(cv).to_csv(out/'validation_scores.csv',index=False)
    pd.DataFrame(splits).to_csv(out/'split_audit.csv',index=False)
    pd.DataFrame(family_selection).to_csv(out/'selected_families.csv',index=False)
    write_json(out/'selection_freeze.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'selected_parameters_sha256':sha(out/'selected_parameters.csv'),
        'selected_families_sha256':sha(out/'selected_families.csv'),
        'note':'All source-only configuration/family selections and final model fits completed before evaluation scoring.'})

    winners={(r['policy'],r['source'],r['horizon_hours'],r['input_set']):r['selected_model'] for r in family_selection}
    metrics=[]
    for (policy,source),d in datasets.items():
        for h in HORIZONS:
            y,_,train,test=split_masks(d,h,CUTOFF)
            for mode in INPUTS:
                x=features[policy,source,mode]
                winner=winners[policy,source,h,mode]
                for split,mask in [('train',train),('test',test)]:
                    actual=y.loc[mask].to_numpy()
                    predictions={'persistence':d.do.loc[mask].to_numpy()}
                    for name in models_for(mode):
                        predictions[name]=predict_frozen(fitted[policy,source,h,mode,name],d,x,mask)
                    for name,pred in predictions.items():
                        metrics.append({'station':source,'policy':policy,'horizon_hours':h,
                            'input_set':mode,'split':split,'model':name,
                            'selected_model':winner,**scores(actual,pred),
                            **detection(actual,pred,threshold=HYPOXIA)})
    metrics=pd.DataFrame(metrics)
    metrics.to_csv(out/'train_test_metrics.csv',index=False)
    write_json(out/'training_progress.json',{'phase':'complete','completed_fits':fit_count,
        'metric_rows':len(metrics)})
    print(f'Completed {fit_count} fits; {len(metrics)} metric rows.',flush=True)
    return metrics



if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args()
    with threadpool_limits(limits=2):run(args.package.resolve())
