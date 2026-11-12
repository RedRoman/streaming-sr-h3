import tempfile
from pathlib import Path
import unittest
import numpy as np
from audit_arrays import array_sha,file_sha,verified_array,verify_unscale,verify_decoder_cast,compare_outputs


class ArrayTests(unittest.TestCase):
    def test_real_shape_file_and_pixels_tamper(self):
        # Full22 latent dimensions, keeping this verification small and meaningful.
        value=np.ones((1,24,7,30,52),np.float32)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'value.npy';np.save(p,value)
            digest=file_sha(p)
            actual=verified_array(p,digest,array_sha(value),value.shape,'float32')
            self.assertEqual(actual.shape,value.shape)
            with self.assertRaisesRegex(ValueError,'content SHA'):
                verified_array(p,digest,'0'*64,value.shape,'float32')
            value.flat[0]=2;np.save(p,value)
            with self.assertRaisesRegex(ValueError,'file SHA'):
                verified_array(p,digest,array_sha(value),value.shape,'float32')

    def test_cast_gradient_scaling_and_parity_gates(self):
        value=np.array([.25,.5,1.],np.float32)
        verify_decoder_cast(value,value.astype(np.float16))
        with self.assertRaisesRegex(ValueError,'actual float16 cast'):
            verify_decoder_cast(value,np.zeros(3,np.float16))
        verify_unscale(value,value/np.float32(1024),nonzero=True)
        with self.assertRaisesRegex(ValueError,'raw/1024'):
            verify_unscale(value,value/np.float32(512))
        compare_outputs(value,value,exact=True)
        with self.assertRaisesRegex(ValueError,'strict output parity'):
            compare_outputs(value+np.float32(1e-5),value,exact=True)
        compare_outputs(value+np.float32(1e-5),value,exact=False)
        with self.assertRaisesRegex(ValueError,'allclose'):
            compare_outputs(value+np.float32(.01),value,exact=False)

if __name__=='__main__':unittest.main()
