"""Match Comfy BaseModel's packed BF16 AV stream IO for raw H3 training.

The condition payload remains FP32, as in the actual official wrapper.  Audio
must be rounded in its carried representation BEFORE the native forward's
conversion back to physical audio; directly rounding sigma_audio*noise differs.
"""
import torch


def official_stream_inputs(video_x_f32, audio_noise_f32, sigma, time_shift_sigma):
    sigma_tensor=torch.as_tensor(sigma,device=video_x_f32.device,dtype=torch.float32).reshape(1)
    timestep=sigma_tensor*1000
    # These operations reproduce BaseModel._apply_model -> MiniMaxH3.forward.
    sigma_from_t=(timestep.flatten()[0]/1000).float().clamp(min=1e-6)
    sigma_audio=time_shift_sigma(sigma_from_t,12,3)
    carry=(sigma_audio/sigma_from_t).to(torch.bfloat16)
    audio_carried=(sigma_tensor.reshape(1,1,1,1)*audio_noise_f32).to(torch.bfloat16)
    physical_audio=audio_carried*carry
    return video_x_f32.to(torch.bfloat16),physical_audio,timestep
