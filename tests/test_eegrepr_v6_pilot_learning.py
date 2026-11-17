import tempfile,unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import learning,config
from experiments.eeg_representation.h1.v6_pilot.panels import match

def panel():
    rng=np.random.default_rng(381);out={};values={}
    for split,start,n in [('train',10,32),('development',20,8),('calibration',30,8),('evaluation',40,8)]:
        x=rng.normal(size=(n,6));y=(2*x[:,0]-x[:,3]+.01*rng.normal(size=n))[:,None]
        values[split]=x
        out[split]=dict(X=x,Y=y[:,None,:],Y_physical=y[:,None,:],future_mask=np.ones((n,1),bool),
            X_physical=x,reference_Y=np.zeros((n,0,1)),reference_Y_physical=np.zeros((n,0,1)),reference_mask=np.zeros((n,0),bool),
            future_id=np.asarray([[f's{start+i//4}-p0000-R{i%4}-score-0000'] for i in range(n)]),
            reference_future_id=np.full((n,0),'',dtype='U1'),structure_id=np.asarray([start+i//4 for i in range(n)]),
            context_id=np.asarray([f's{start+i//4}-p0000-R{i%4}' for i in range(n)]),regime=np.asarray([f'R{i%4}' for i in range(n)]))
    return out,values,dict(target_fingerprint='engineering',risk_contract={'target':'learnable'})

class PilotLearningTests(unittest.TestCase):
    def test_actual_candidate_learns_and_eval_never_selects(self):
        p,x,m=panel();c=config.resolve()
        with tempfile.TemporaryDirectory() as temp:
            a=learning.ridge(p,x,m,Path(temp)/'a',c)
            pred=learning.numeric(Path(temp)/'a/predictions.npz')
            self.assertLess(np.mean((pred['evaluation_prediction']-p['evaluation']['Y'][:,0])**2),.01)
            changed=deepcopy(p);changed['evaluation']['Y'][:]=1e4;changed['calibration']['Y'][:]=-3e4
            b=learning.ridge(changed,x,m,Path(temp)/'b',c)
            self.assertEqual(a['selected_id'],b['selected_id'])
            za=learning.numeric(Path(temp)/'a/state.npz');zb=learning.numeric(Path(temp)/'b/state.npz')
            for key in za:np.testing.assert_array_equal(za[key],zb[key])
            self.assertTrue(a['actually_fitted_state_dependent'])
            self.assertIn('evaluation_candidate_prediction',pred)

    def test_mean_selected_keeps_trained_state_candidate(self):
        p,x,m=panel();p['development']['Y'][:]=p['train']['Y'][:,0].mean(0)
        with tempfile.TemporaryDirectory() as temp:
            a=learning.ridge(p,x,m,temp,config.resolve())
            self.assertEqual(a['selected_id'],'train_mean')
            self.assertTrue(next(r for r in a['candidates'] if r['id']==a['predeclared_candidate_id'])['actually_trained'])
            z=learning.numeric(Path(temp)/'predictions.npz')
            self.assertGreater(np.std(z['evaluation_candidate_prediction']),0)

    def test_resume_rejects_changed_inputs_or_output(self):
        p,x,m=panel();c=config.resolve()
        with tempfile.TemporaryDirectory() as temp:
            learning.ridge(p,x,m,temp,c)
            self.assertEqual(learning.ridge(p,x,m,temp,c,resume=True)['status'],'complete')
            x['evaluation'][0,0]+=1
            with self.assertRaises(ValueError):learning.ridge(p,x,m,temp,c,resume=True)

    def test_separate_pilot_cannot_launch_confirmation(self):
        c=config.resolve();self.assertEqual(c['max_new_invocations'],160);self.assertEqual(len(config.jobs()),60)
        for changes in ({'workers':4},{'max_new_invocations':161},{'h2_l':128},{'neural_max_steps':501}):
            with self.assertRaises(ValueError):config.resolve(changes)

    def test_matched_normalization_training_only(self):
        p,x,m=panel();m['normalization']={'original':'preserved'}
        ids={s:list(v['context_id']) for s,v in p.items()}
        a,ma=match(p,ids,m);p['evaluation']['Y_physical'][:]+=100
        b,mb=match(p,ids,m)
        self.assertEqual(ma['normalization'],mb['normalization'])
        self.assertEqual(ma['normalization']['fit_split'],'train')

if __name__=='__main__':unittest.main()
