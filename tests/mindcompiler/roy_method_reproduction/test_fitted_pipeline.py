"""Lifecycle, leakage-by-construction, and metric tests + replay equivalence."""
import numpy as np
import pytest
from fmri2img.mindcompiler.roy_method_reproduction.fitted_pipeline import (
    FittedPipeline, LifecycleError, State)
from fmri2img.mindcompiler.roy_method_reproduction.metrics import per_voxel_report

def _data(seed=0, n=40, p=8, q=6):
    rng=np.random.default_rng(seed)
    X=rng.standard_normal((n,p)); B=rng.standard_normal((p,2))@rng.standard_normal((2,q))
    Y=X@B+0.1*rng.standard_normal((n,q)); return X,Y

def _pipe():
    return FittedPipeline().resolve_policies()

# --- lifecycle ordering ---------------------------------------------------
def test_predict_before_fit_raises():
    with pytest.raises(LifecycleError, match="predict before"):
        FittedPipeline().predict(np.zeros((2,8)))

def test_select_before_preprocessing_raises():
    X,Y=_data(); p=_pipe()
    with pytest.raises(LifecycleError, match="select from"):
        p.select(X,Y)

def test_evaluate_before_final_fit_raises():
    X,Y=_data(); p=_pipe().fit_preprocessing(X[:24],Y[:24]).select(X[24:32],Y[24:32])
    with pytest.raises(LifecycleError, match="evaluate_test from"):
        p.evaluate_test(X[32:],Y[32:])

def test_second_test_evaluation_raises():
    X,Y=_data()
    p=_pipe().fit_preprocessing(X[:24],Y[:24]).select(X[24:32],Y[24:32]).fit_final()
    p.evaluate_test(X[32:],Y[32:])
    # the state guard (TEST_EVALUATED) is the real one-shot protection
    with pytest.raises(LifecycleError, match="evaluate_test from|already evaluated"):
        p.evaluate_test(X[32:],Y[32:])

def test_full_lifecycle_seals():
    X,Y=_data()
    p=_pipe().fit_preprocessing(X[:24],Y[:24]).select(X[24:32],Y[24:32]).fit_final()
    p.evaluate_test(X[32:],Y[32:]); s=p.seal()
    assert s["state"]=="SEALED" and s["policy_name"].startswith("HISTORICAL")

def test_unresolved_policy_rejected():
    p=FittedPipeline(source_centering="magic")
    with pytest.raises(LifecycleError, match="unresolved/implicit"):
        p.resolve_policies()

def test_mismatched_dims_rejected():
    X,Y=_data(p=8); p=_pipe().fit_preprocessing(X[:24],Y[:24]).select(X[24:32],Y[24:32]).fit_final()
    with pytest.raises(LifecycleError, match="source dim"):
        p.evaluate_test(np.zeros((8,5)),Y[32:])

# --- leakage by construction ----------------------------------------------
def test_preprocessing_uses_train_statistics_only():
    """Centering must use TRAIN mean; validation/test are transformed by it."""
    X,Y=_data(); Xtr=X[:24]+10.0
    p=_pipe().fit_preprocessing(Xtr,Y[:24])
    # the object stores train-centered arrays; transforming a shifted test set
    # must NOT re-center it to its own mean
    out=p._sx.transform(X[24:]+50.0)
    assert not np.allclose(out.mean(0),0,atol=1e-6)

# --- metrics --------------------------------------------------------------
def test_metric_all_finite():
    rng=np.random.default_rng(1); Yt=rng.standard_normal((20,5)); Yp=Yt+0.1*rng.standard_normal((20,5))
    r=per_voxel_report(Yt,Yp); assert r.finite==5 and r.finite_fraction==1.0 and r.target_constant==0

def test_metric_constant_target_and_prediction_counted():
    Yt=np.random.default_rng(2).standard_normal((20,4)); Yp=Yt.copy()
    Yt[:,0]=3.0        # constant target
    Yp[:,1]=7.0        # constant prediction
    r=per_voxel_report(Yt,Yp)
    assert r.target_constant>=1 and r.prediction_constant>=1
    assert r.nan>=1  # at least the constant-target voxel is undefined

def test_metric_all_undefined():
    Yt=np.ones((10,3)); Yp=np.random.default_rng(3).standard_normal((10,3))
    r=per_voxel_report(Yt,Yp)
    assert r.finite==0 and r.finite_fraction==0.0 and r.target_constant==3
    assert not r.passes(0.90)

def test_metric_threshold_pass_fail():
    rng=np.random.default_rng(4); Yt=rng.standard_normal((20,10)); Yp=Yt+0.05*rng.standard_normal((20,10))
    assert per_voxel_report(Yt,Yp).passes(0.90)
    Yt[:,:5]=1.0  # half constant -> finite fraction 0.5
    assert not per_voxel_report(Yt,Yp).passes(0.90)
