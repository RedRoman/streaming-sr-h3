"""result.json written before the fix let the sampler's 'mode' overwrite the run mode; the folder name is authoritative."""
import glob, json
n = 0
for f in glob.glob('/cache/zihaohe/streaming_sr0/results/step0_blocks/*/*/*/result.json'):
    folder_mode = f.split('/')[-4]
    r = json.load(open(f))
    if r['mode'] != folder_mode:
        r['sampler_mode'], r['mode'], r['mode_label_fixed'] = r['mode'], folder_mode, True
        json.dump(r, open(f, 'w'), indent=1); n += 1
print('fixed', n)
