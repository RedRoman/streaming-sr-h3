"""Verify the completed pilot against its immutable preregistered budget."""
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text());digest=hashlib.sha256((a.run/'protocol.json').read_bytes()).hexdigest();rows=[]
    assert protocol['version']=='native_io_v2'
    root=Path(__file__).resolve().parents[2]
    legacy=json.loads((root/'results/h3_setup/native_sr_pilot400/protocol.json').read_text())
    for key in ['seed','updates_each','controls','lr','optimizer','training_sigma','train_cases','validation_cases','validation_sigmas','validation_schedule','readout','lora','guide_projection','context_sha256','model_sha256','train_manifest_sha256','val_manifest_sha256','schedule']:
        assert protocol[key]==legacy[key],('Unintended non-IO protocol drift',key)
    math_path=root/'results/h3_setup/native_math_alignment_v2/result.json'
    math=json.loads(math_path.read_text());assert math['native_io_v2_audit_passed']
    assert hashlib.sha256(math_path.read_bytes()).hexdigest()==protocol['official_wrapper_math_backward_audit_sha256']
    assert protocol['native_io_helper_sha256']==math['native_io_helper_sha256']
    assert all(r['v_native_io_v2_vs_official_wrapper']['max_abs']==0 and r['training_x0_vs_official_oneEuler_output']['max_abs']<2e-5 for r in math['conditions'])
    backward=math['native_io_v2_real_backward'];assert backward['guide_input_grad_norm']>0 and backward['lora_grad_norm']>0 and backward['guide_projection_grad_norm']>0 and backward['base_weight_grad_count']==0
    parity=json.loads((a.run/'zero_adapter_parity.json').read_text());assert all(value==0 for value in parity.values())
    initial=[]
    validations=list(map(json.loads,(a.run/'validation.jsonl').read_text().splitlines()))
    for mode in ['lora_only','guide_only','joint']:
        result=json.loads((a.run/mode/'result.json').read_text());assert result['status']=='complete' and result['updates_completed']==400 and result['protocol_sha256']==digest
        assert result['base_parameter_versions_unchanged'] and result['base_weight_grad_count']==0
        logged=list(map(json.loads,(a.run/mode/'train.jsonl').read_text().splitlines()))
        # Append-only logs can preserve a failed/resumed tail; last instance of
        # each update is the actual resumed trajectory, with samefixednoise.
        updates={entry['update']:entry for entry in logged};assert sorted(updates)==list(range(400))
        for scheduled in protocol['schedule']:
            row=updates[scheduled['update']]
            assert row['mode']==mode and row['base_weight_grad_count']==0 and row['adapter_grad_norm_before_clip']>0 and row['guide_input_grad_norm']>0
            assert all(row[key]==scheduled[key] for key in ['case','sigma','noise_seed'])
            assert row['case'] in protocol['train_cases'] and row['case'] not in protocol['validation_cases']
        val={v['update']:v for v in validations if v['mode']==mode};assert sorted(val)==[0,100,200,300,400]
        for item in val.values():
            assert len(item['rows'])==12
            assert {r['case'] for r in item['rows']}==set(protocol['validation_cases']) and {r['sigma'] for r in item['rows']}=={.2,.5,.8}
        initial.append(val[0]['mean_velocity_mse'])
        for case in protocol['validation_cases']:
            sample=json.loads((a.run/'readouts'/mode/case/'sample_result.json').read_text())
            assert sample['status']=='complete' and sample['split']=='val' and not sample['HR_condition']
            assert sample['steps_model_evaluations']==49 and sample['sigma_points']==50 and sample['seed']==101 and sample['cfg']==1 and sample['input_prefix']==[0,21]
        rows.append({'mode':mode,'unique_updates':400,'append_only_log_rows':len(logged),'trainable_parameters':result['trainable_parameters'],'fixed_validation_initial':val[0]['mean_velocity_mse'],'fixed_validation_final400':val[400]['mean_velocity_mse'],'base_freeze_passed':True,
                     'guide_input_grad_norm_range':[min(v['guide_input_grad_norm'] for v in updates.values()),max(v['guide_input_grad_norm'] for v in updates.values())]})
    assert max(initial)-min(initial)<1e-12,initial
    result={'passed':True,'protocol_sha256':digest,'zero_adapter_parity':parity,'rows':rows,'native_io_v2_official_parity_and_backward':math,'legacy_to_v2_schedule_equal':True,'checks':['actualofficialBF16streamIOparity0, realbackwardnonzero, frozenbasegrad0','same400case/sigma/noiseseeds acrosscontrols','validationneveroptimizerinput','fixedfinal400, no checkpointselection','base tensorversionsunchanged andgrad0','allnative49samplersLRonly, samepromptcontext/prefix/seed/CFG']}
    (a.run/'protocol_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
