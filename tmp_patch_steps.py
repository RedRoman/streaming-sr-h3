import pathlib, sys
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/cache/zihaohe/streaming_sr0/code/h3_comfy.py')
s = p.read_text()

old = """        T = guide.shape[2]
        h16, w16 = guide.shape[3], guide.shape[4]"""
new = """        # the step count follows the sigma schedule: a turbo LoRA runs the same sampler on 8 points, not 49
        n_steps = len(SIGMAS) - 1
        # innerDC's projection points were chosen on the 49-step grid; keep their relative position
        points = sorted({min(n_steps - 1, max(0, round(i * n_steps / 49))) for i in POINTS})
        T = guide.shape[2]
        h16, w16 = guide.shape[3], guide.shape[4]"""
assert old in s
s = s.replace(old, new)

s = s.replace("            if state['calls'] in [0, *POINTS]:", "            if state['calls'] in [0, *points]:")
s = s.replace("            if index not in POINTS:", "            if index not in points:")
s = s.replace("""                samples = comfy.sample.sample(current, comfy.sample.prepare_noise(latent, seed), 49, 1., 'euler', 'simple', positive, [], latent,
                                              seed=seed, disable_pbar=True, sigmas=SIGMAS)""",
              """                samples = comfy.sample.sample(current, comfy.sample.prepare_noise(latent, seed), n_steps, 1., 'euler', 'simple', positive, [], latent,
                                              seed=seed, disable_pbar=True, sigmas=SIGMAS)""")
s = s.replace("""                z = comfy.sample.prepare_noise(latent, seed) if noise is None else noise
                samples = comfy.sample.sample(current, z, 49, 1., 'euler', 'simple', positive, [], latent,""",
              """                z = comfy.sample.prepare_noise(latent, seed) if noise is None else noise
                samples = comfy.sample.sample(current, z, n_steps, 1., 'euler', 'simple', positive, [], latent,""")
s = s.replace("        assert state['calls'] == 49 and torch.isfinite(video).all() and torch.isfinite(audio).all()",
              "        assert state['calls'] == n_steps and torch.isfinite(video).all() and torch.isfinite(audio).all()")
s = s.replace("            assert state['hooks'] == 49 and [e['index'] for e in events] == POINTS",
              "            assert state['hooks'] == n_steps and [e['index'] for e in events] == points")
s = s.replace("""        return {'video': video.cpu(), 'audio': audio.cpu()}, {'mode': mode, 'seed': seed, 'steps': 49, 'cfg': 1.,""",
              """        return {'video': video.cpu(), 'audio': audio.cpu()}, {'mode': mode, 'seed': seed, 'steps': n_steps, 'cfg': 1.,""")
p.write_text(s)
import ast; ast.parse(s)
print('patched', p)
