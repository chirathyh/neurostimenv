import copy,unittest
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import analysis,config
from experiments.eeg_representation.h1.v6.information import summarize

class PairedPilotTests(unittest.TestCase):
    def row(self,method,prediction):
        rng=np.random.default_rng(9);y=rng.normal(size=(8,8,4))
        ids=np.asarray([f's{40+i//4}-p0000-R{i%4}' for i in range(8)])
        panel=dict(structure_id=np.repeat([40,41],4),regime=np.tile(['R0','Ralpha','Rbeta1','Rbeta2'],2),
                   context_id=ids,future_id=np.asarray([[f'{cid}-score-{j:04d}' for j in range(8)] for cid in ids]))
        n,_=summarize(y,np.full((8,4),prediction),np.zeros((8,4)),panel,method_id=method)
        return dict(job='A0__T0_l0_d128',acquisition='A0',target='T0_l0_d128',h_ms=0,duration_ms=128,
                    method=method,population='matched',full_numeric=n,circuits=2,
                    reference_risk_physical=n['reference_mse'],error_baseline100=n['error_baseline100'],
                    relative_mse_reduction=n['relative_mse_reduction_pp'])

    def test_paired_contrast_sign_and_mc(self):
        a=self.row('eeg_full',.1);b=self.row('eeg_short',.5)
        r=analysis.comparison(a,b,'history',config.resolve())
        self.assertAlmostEqual(r['paired_difference'],b['error_baseline100']-a['error_baseline100'])
        self.assertEqual(r['uncertainty']['status'],'deferred')
        self.assertIn('conditional_mc_gain_difference_se',r['paired_descriptive'])
        self.assertNotIn('within_tolerance',r['paired_descriptive']['gain_difference'])

    def test_paired_rows_require_identical_population_and_future_ids(self):
        a=self.row('a',0);b=self.row('b',1)
        b['full_numeric']['future_ids'][0][0]='unpaired'
        with self.assertRaisesRegex(ValueError,'future_ids'):analysis.comparison(a,b,'history',config.resolve())
        b=self.row('b',1);b['reference_risk_physical']*=2
        with self.assertRaisesRegex(ValueError,'different training'):analysis.comparison(a,b,'acquisition',config.resolve())
        b=self.row('b',1);b['target']='T0_l10_d128'
        with self.assertRaisesRegex(ValueError,'same physical target'):analysis.comparison(a,b,'acquisition',config.resolve())

if __name__=='__main__':unittest.main()

class ArtifactPipelineTests(unittest.TestCase):
    def test_bounded_fits_and_readouts_score_numeric_artifacts(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tests.test_eegrepr_v6_models import panel_fixture
        from experiments.eeg_representation.h1.v6_pilot import learning
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for cohort in ('matched','broad'):
                p,m,_,job=panel_fixture(root/cohort,length=64,futures=8)
            def bank(c,stage,panel,*args):
                return {'pca_B32':{'values':{s:v['X'][:,:8] for s,v in panel.items()},'metadata':{'projection':'d8','payload_bits':32,'purpose':'engineering'}}}
            with patch.object(analysis.banks,'build',side_effect=bank):
                fit=analysis.fit_ordinary(config.resolve(),root,'unused_engineering_archive',[job],device='cpu')
            self.assertEqual(fit['matched']['status'],'complete')
            tables=analysis.score_models(config.resolve(),root,[job])
            self.assertEqual(len(tables['h3_retention']),2)
            for row in tables['h3_retention']:
                self.assertEqual(row['projection'],'d8')
                self.assertIsNotNone(row['carrier_cross_entropy_bits'])
            self.assertEqual(tables['h4_targets'][0]['physical_units'],'V')
            self.assertEqual(tables['h4_targets'][0]['risk_units'],'V^2')
            # Exercise the actual scoring/report boundary, including mean fallback.
            from experiments.eeg_representation.h1.v6_pilot import reporting
            normalized=reporting.normalize_results(dict(tables=tables))
            self.assertEqual(len(normalized['tables']['h1_information']),len(tables['h1_information']))
            self.assertEqual(reporting.normalize_results(normalized),normalized)
            # Scoring may not silently accept a modified prediction artifact.
            path=root/'matched/learners'/job/'eeg_short/predictions.npz'
            data=learning.numeric(path);data['evaluation_prediction'][0,0]+=1
            np.savez_compressed(path,**data)
            with self.assertRaisesRegex(ValueError,'checksum'):analysis.score_models(config.resolve(),root,[job])
