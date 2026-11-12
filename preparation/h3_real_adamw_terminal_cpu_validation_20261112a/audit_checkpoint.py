"""Independent safetensors and SGD-array audit using stdlib/NumPy, no Torch."""
import json
import math
from pathlib import Path
import struct

import numpy as np

from audit_arrays import require, file_sha, array_sha, verify_reduction


class SafeAdapter:
    def __init__(self,path,expected_sha,expected_shapes,expected_metadata,provenance=True):
        self.path=Path(path)
        require(self.path.suffix=='.safetensors' and self.path.stat().st_size <= 512<<20,'unexpected checkpoint size/type')
        require(file_sha(self.path)==expected_sha,'checkpoint file SHA mismatch')
        with self.path.open('rb') as stream:
            raw=stream.read(8);require(len(raw)==8,'truncated safetensors length')
            size=struct.unpack('<Q',raw)[0]
            require(0 < size <= 8<<20,'unbounded safetensors header')
            header=stream.read(size);require(len(header)==size,'truncated safetensors header')
        def unique(pairs):
            result={}
            for key,value in pairs:
                require(key not in result,'duplicate checkpoint JSON key');result[key]=value
            return result
        self.header=json.loads(header,object_pairs_hook=unique)
        metadata=self.header.pop('__metadata__',{})
        if provenance:
            require(set(metadata)=={'openh3_provenance'},'unexpected checkpoint metadata keys')
            require(json.loads(metadata['openh3_provenance'],object_pairs_hook=unique)==expected_metadata,'checkpoint provenance differs')
        else:
            require(metadata==expected_metadata,'optimizer checkpoint metadata differs')
        require(set(self.header)==set(expected_shapes),'checkpoint must contain exact adapter key set')
        self.offset=8+size
        spans=[]
        for name,shape in expected_shapes.items():
            row=self.header[name]
            require(set(row)=={'dtype','shape','data_offsets'} and row['dtype']=='F32' and row['shape']==list(shape),
                'checkpoint exact FP32 shape differs: '+name)
            lo,hi=row['data_offsets']
            require(type(lo) is int and type(hi) is int and 0<=lo<hi and hi-lo==math.prod(shape)*4,
                'checkpoint byte offsets/length differ')
            spans.append((lo,hi))
        cursor=0
        for lo,hi in sorted(spans):
            require(lo==cursor,'overlapping/gapped checkpoint tensors');cursor=hi
        require(self.offset+cursor==self.path.stat().st_size,'checkpoint trailing/truncated data')

    def load(self,name):
        row=self.header[name];lo,hi=row['data_offsets']
        with self.path.open('rb') as stream:
            stream.seek(self.offset+lo);data=stream.read(hi-lo)
        value=np.frombuffer(data,dtype='<f4').reshape(row['shape'])
        require(np.isfinite(value).all(),'nonfinite checkpoint tensor: '+name)
        return value


def update_tensor(before,gradient,after,row):
    require(before.dtype==gradient.dtype==after.dtype==np.float32 and before.shape==gradient.shape==after.shape,
            'SGD array shape/dtype differs')
    require(all(np.isfinite(x).all() for x in (before,gradient,after)),'nonfinite SGD values')
    require(row['before_sha256']==array_sha(before) and row['gradient_unscaled_sha256']==array_sha(gradient)
        and row['after_sha256']==array_sha(after) and row['expected_after_sha256']==array_sha(after)
        and row['exact_SGD_arithmetic'] is True and row['all_finite'] is True,
        'producer same-device SGD exact expected/actual bytes differ')
    # CPU NumPy split mul/add is not claimed to reproduce a backend FMA.
    # Only algebraically unambiguous coordinates are a new independent gate.
    alpha=np.float32(-1e-4)
    split=before+gradient*alpha
    fused=(before.astype(np.float64)+gradient.astype(np.float64)*np.float64(alpha)).astype(np.float32)
    zero_grad=gradient==0
    zero_before=before==0
    require(array_sha(after[zero_grad])==array_sha(before[zero_grad]),'zero-gradient parameter changed')
    # Initial B is +0, and zero-gradient -0 signs are handled by the stricter
    # preceding unchanged-byte gate. Nonzero products have unambiguous FP32 bytes.
    product_mask=zero_before & ~zero_grad
    require(array_sha(after[product_mask])==array_sha((gradient*alpha)[product_mask]),'zero-initialized SGD product differs')
    delta=(after-before).astype(np.float64)
    verify_reduction(row['update_l2'],float(np.linalg.norm(delta)),'actual update norm')
    verify_reduction(row['update_rms'],float(np.sqrt(np.mean(delta*delta))),'actual update RMS')
    require(row['changed_coordinates']==int(np.count_nonzero(delta)),'actual update coordinate count differs')
    nontrivial=~(zero_grad|zero_before)
    return {'coordinates':int(before.size),'independent_zero_gradient_coordinates':int(zero_grad.sum()),
        'independent_zero_before_product_coordinates':int(product_mask.sum()),
        'nontrivial_coordinates':int(nontrivial.sum()),
        'descriptive_nontrivial_split_matches':int(np.count_nonzero((split==after)&nontrivial)),
        'descriptive_nontrivial_fused64_matches':int(np.count_nonzero((fused==after)&nontrivial)),
        'descriptive_split_max_abs':float(np.max(np.abs(split.astype(np.float64)-after.astype(np.float64)))),
        'descriptive_fused64_max_abs':float(np.max(np.abs(fused.astype(np.float64)-after.astype(np.float64)))),
        'nontrivial_candidate_union_is_acceptance_gate':False,
        'scope':'Saved same-device expected SHA is required. CPU candidates are descriptive outside algebraically unambiguous coordinates; no NPU FMA claim.'}
