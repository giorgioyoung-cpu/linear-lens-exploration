"""Numerical checks against explicit weighted least squares; requires only NumPy."""
import unittest
import numpy as np
from lens_core import stats,fit_stats,fit_paired,error

class AffineChecks(unittest.TestCase):
    def test_weighted_objectives(self):
        rng=np.random.default_rng(7)
        records=[(rng.normal(size=(n,3))+2,rng.normal(size=(n,3))-4) for n in (2,5,7)]
        for same in (False,True):
            xs=[];ys=[];weights=[]
            for u,y in records:
                pairs=[(i,j) for j in range(len(u)) for i in (range(j,len(u)) if not same else [j])]
                for i,j in pairs:
                    xs.append(u[j]);ys.append(y[i]);weights.append(1/len(records)/len(pairs))
            x,y,w=np.array(xs),np.array(ys),np.array(weights)
            design=np.column_stack((x,np.ones(len(x))))
            expected=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=None)[0]
            s=stats(records,same);A,b,_=fit_stats(s)
            np.testing.assert_allclose(A,expected[:-1].T,atol=1e-11)
            np.testing.assert_allclose(b,expected[-1],atol=1e-11)
            np.testing.assert_allclose(error(A,b,s),np.sum(w*np.sum((y-x@A.T-b)**2,axis=1)),atol=1e-11)
            # Also test arbitrary predictions, not only the minimizing map.
            B=rng.normal(size=(3,3));c=rng.normal(size=3)
            np.testing.assert_allclose(error(B,c,s),np.sum(w*np.sum((y-x@B.T-c)**2,axis=1)),atol=1e-10)

    def test_paired_explicit_and_translation(self):
        rng=np.random.default_rng(19)
        for N,d in ((9,3),(3,7)):
            u=rng.normal(size=(N,4,d))+3;y=rng.normal(size=(N,4,d))-2
            A,b,ranks=fit_paired(u,y)
            As=[];bs=[]
            for j in range(4):
                xc=u[:,j]-u[:,j].mean(axis=0)
                for i in range(j,4):
                    yc=y[:,i]-y[:,i].mean(axis=0)
                    B=(np.linalg.pinv(xc,rcond=1e-5)@yc).T
                    As.append(B);bs.append(y[:,i].mean(axis=0)-B@u[:,j].mean(axis=0))
            np.testing.assert_allclose(A,np.mean(As,axis=0),atol=1e-11)
            np.testing.assert_allclose(b,np.mean(bs,axis=0),atol=1e-11)
            shifted,_,_=fit_paired(u+11,y-9)
            np.testing.assert_allclose(A,shifted,atol=1e-11)
            self.assertTrue(all(r['rank']<=min(N-1,d) for r in ranks))

    def test_known_affine_and_degenerate(self):
        rng=np.random.default_rng(5);u=rng.normal(size=(50,3))+2
        B=rng.normal(size=(3,3));c=np.array([1.,2.,3.]);y=u@B.T+c
        A,b,_=fit_stats(stats([(u,y)],True))
        np.testing.assert_allclose(A,B,atol=1e-11);np.testing.assert_allclose(b,c,atol=1e-11)
        A,b,r=fit_stats(stats([(np.ones((5,3)),np.full((5,3),4.))],True))
        np.testing.assert_allclose(A,0);np.testing.assert_allclose(b,4);self.assertEqual(r['rank'],0)
        A,b,_=fit_paired(np.ones((4,3,2)),np.full((4,3,2),4.))
        np.testing.assert_allclose(A,0);np.testing.assert_allclose(b,4)

class MultiLayerChecks(unittest.TestCase):
    def test_counting_alignment(self):
        from evaluate_concepts import counting_positions
        # Prompt ends at zero-based 2; response tokens 7,8; EOS=99.
        self.assertEqual(counting_positions(3,[1,2,3,7,8,99],{99}),[2,3,4])
        self.assertEqual(counting_positions(3,[1,2,3,7,8],{99}),[2,3,4])
        self.assertEqual(counting_positions(3,[1,2,3,99],{99}),[2])

    def test_layer_dispatch_preserves_options(self):
        import tempfile, json, sys
        from pathlib import Path
        from unittest.mock import patch
        from lens_core import dispatch_layers, layer_runs
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'layers.json').write_text(json.dumps({'status':'complete',
                'layers':{'12':'layer_12','4':'layer_04'}}))
            self.assertEqual([l for l,_ in layer_runs(root)],[4,12])
            argv=['fit_jlens.py','--run-dir',folder,'--fit','--prompts','8']
            with patch.object(sys,'argv',argv), patch('subprocess.run') as run:
                dispatch_layers(root)
                self.assertEqual(run.call_count,2)
                for call, layer in zip(run.call_args_list,['layer_04','layer_12']):
                    command=call.args[0]
                    self.assertEqual(Path(command[command.index('--run-dir')+1]).resolve(),(root/layer).resolve())
                    self.assertEqual(command[-3:],['--fit','--prompts','8'])
                    self.assertTrue(call.kwargs['check'])
            (root/'layers.json').write_text(json.dumps({'status':'incomplete','layers':{}}))
            with self.assertRaises(ValueError): layer_runs(root)

class SnapshotChecks(unittest.TestCase):
    def test_prefix_averages_and_evaluator_loading(self):
        import tempfile, json
        from pathlib import Path
        from fit_jlens import save_snapshot
        from lens_core import load_suite, sha
        rng=np.random.default_rng(81)
        jacobians=rng.normal(size=(118,3,3))
        means=rng.normal(size=(118,3))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'metadata.json').write_text('{}')
            out=root/'jlens_fit_test'; out.mkdir()
            config={'model_name':'synthetic','model_revision':None,
                    'source_layer':12,'target_layer':24,
                    'collection_metadata_sha256':sha(root/'metadata.json'),
                    'selected_training_files':[{'filename':str(i)} for i in range(118)],
                    'requested_prompts':118,'format_version':1}
            fit=root/'affine_fit_test'; fit.mkdir()
            report=dict(config, format_version=2, centered=True, activation_run=str(root))
            (fit/'fit_report.json').write_text(json.dumps(report))
            arrays={name+'_A':np.eye(3) for name in ('pooled','same','paired')}
            arrays.update({name+'_mean':np.zeros(3) for name in ('pooled','same')})
            np.savez(fit/'lenses.npz',**arrays)
            total=np.zeros((3,3)); mean_total=np.zeros(3)
            for n,(J,m) in enumerate(zip(jacobians,means),1):
                total+=J; mean_total+=m
                if n not in (8,32,64,118): continue
                saved=save_snapshot(out,total,mean_total,n,config)
                jr=json.loads((saved/'fit_report.json').read_text())
                self.assertEqual(jr['n_prompts'],n)
                self.assertEqual(len(jr['selected_training_files']),n)
                self.assertEqual(jr['parent_requested_prompts'],118)
                self.assertEqual(jr['status'],'complete')
                np.testing.assert_allclose(np.load(saved/'mean_comparison_jlens/training_target_mean.npy'),means[:n].mean(0))
                _,loaded,maps,_=load_suite(fit,root,out,n)
                np.testing.assert_allclose(maps['jlens'],jacobians[:n].mean(0))
                self.assertEqual(loaded['jlens_comparison']['passages'],n)
                # A replay after interruption preserves the published snapshot.
                self.assertEqual(save_snapshot(out,total,mean_total,n,config),saved)
            with self.assertRaises(FileNotFoundError): load_suite(fit,root,out,16)
            with self.assertRaises(ValueError): save_snapshot(out,total+1,mean_total,118,config)


class PauseChecks(unittest.TestCase):
    def test_pause_and_resume_boundaries(self):
        from fit_jlens import stage_limit
        self.assertEqual(list(range(118))[0:stage_limit(118,32)],list(range(32)))
        self.assertEqual(list(range(118))[32:stage_limit(118,64)],list(range(32,64)))
        self.assertEqual(stage_limit(118,None),118)
        for n in (0,119):
            with self.assertRaises(ValueError): stage_limit(118,n)

    def test_stability_report_known_changes(self):
        import tempfile, json
        from pathlib import Path
        from fit_jlens import write_stability_report
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for n,scale in [(8,1),(32,2),(64,2)]:
                target=root/'snapshots'/f'prompts_{n:04d}'
                target.mkdir(parents=True)
                np.savez(target/'linear_lens.npz',A=scale*np.eye(3))
            rows=write_stability_report(root)
            self.assertAlmostEqual(rows[0]['relative_frobenius_change'],1)
            self.assertAlmostEqual(rows[0]['matrix_cosine'],1)
            self.assertAlmostEqual(rows[0]['norm_ratio'],2)
            self.assertAlmostEqual(rows[1]['relative_frobenius_change'],0)
            self.assertEqual(len(json.loads((root/'stability.json').read_text())['comparisons']),2)

if __name__=='__main__': unittest.main()


