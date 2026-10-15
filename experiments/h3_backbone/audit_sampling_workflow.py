"""Numerical parity with official EmptyAV/Noise/BasicGuider/CustomAdvanced nodes."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from safetensors.torch import load_file,save_file


def main():
    p=argparse.ArgumentParser();p.add_argument('--comfy',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--context',type=Path,required=True);p.add_argument('--case',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.comfy));from comfy.cli_args import args as cargs
    cargs.reserve_vram=20.0
    import comfy.sd,comfy.sample,comfy.samplers,comfy.utils
    from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
    from comfy_extras.nodes_custom_sampler import BasicScheduler,BasicGuider,SamplerCustomAdvanced,Noise_RandomNoise
    comfy.utils.PROGRESS_BAR_ENABLED=False
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.5);torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    manifest=json.loads((a.case/'guide_manifest.json').read_text());previous=json.loads((a.case/'sample_result.json').read_text())
    latent=EmptyMiniMaxH3LatentAV.execute(width=256,height=256,length=manifest['target_frames']).result[0]
    cache=torch.load(a.context,map_location='cpu',weights_only=True)
    positive=[[cache['prompt_embeds'][None],{'minimax_token_tags':cache['text_token_tags'],'minimax_keyframes':[{'resolved_frame_index':0,'latent':load_file(str(a.case/'guide.safetensors'))['guide']}]}]]
    model=comfy.sd.load_diffusion_model(str(a.model),model_options={'dtype':torch.bfloat16})
    guider=BasicGuider.execute(model,positive).result[0]
    assert guider.cfg==1.0
    noise=Noise_RandomNoise(101);native_noise=noise.generate_noise(latent);direct_noise=comfy.sample.prepare_noise(latent['samples'],101)
    noise_diff=[float((x-y).abs().max()) for x,y in zip(native_noise.unbind(),direct_noise.unbind())]
    sigmas=torch.tensor(previous['sigma_grid'],dtype=torch.float32)
    node_simple=BasicScheduler.execute(model,'simple',49,1.0).result[0]
    with torch.no_grad():
        output=SamplerCustomAdvanced.execute(noise,guider,comfy.samplers.sampler_object('euler'),sigmas,latent).result[0]['samples']
    old=load_file(str(a.case/'sampled_av_latents.safetensors'))
    video,audio=output.unbind();save_file({'video':video.cpu().contiguous(),'audio':audio.cpu().contiguous()},str(a.case/'official_workflow_av_latents.safetensors'))
    result={'status':'complete','scope':'Exact official sampling nodes with identical cached actual AddGuide output, prompt, seed101, CFG1/Euler and exact normal50 sigma points; no Turbo LoRA, no model patches',
            'actual_nodes':['EmptyMiniMaxH3LatentAV.execute','Noise_RandomNoise.generate_noise','BasicGuider.execute','SamplerCustomAdvanced.execute'],
            'guide_encoding':'Actual official AddGuide was separately parity-verified; reuse its exact saved output to keep VAE and40GBDiT in separate processes',
            'noise_maxabs_per_stream':noise_diff,'video_maxabs_vs_direct_sampler':float((video.cpu()-old['video']).abs().max()),'audio_maxabs_vs_direct_sampler':float((audio.cpu()-old['audio']).abs().max()),
            'sigma_maxabs_vs_official_BasicScheduler_simple49':float((sigmas-node_simple).abs().max()),
            'cfg':guider.cfg,'model_sampling_shifts':[model.model.model_sampling.shift,model.model.model_sampling.audio_shift],
            'video_head_shape':list(model.model.diffusion_model.final_layer.video_out.weight.shape),'audio_head_shape':list(model.model.diffusion_model.final_layer.audio_out.weight.shape),
            'head_note':'Single native velocity head (96 video/32 audio), no PDD/Turbo head bank',
            'seconds':time.perf_counter()-began,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    (a.case/'official_workflow_parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
