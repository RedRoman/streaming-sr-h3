import pathlib, sys
p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/cache/zihaohe/streaming_sr0/code/train_stage1.py')
s = p.read_text()

s = s.replace("""    p.add_argument('--guide-key', default='guide',""",
"""    p.add_argument('--perc', type=float, default=0.0,
                   help="weight on an LPIPS term computed on decoded pixels. Velocity MSE is only weakly "
                        "related to what we are judged on: training with a clean range estimate cut it by "
                        "39%% and bought 0.0184 LPIPS, and switching the training data from generated to "
                        "real footage moved LPIPS by 0.0003. Real-ESRGAN optimises perceptual quality "
                        "directly and beats us on it, so this optimises it directly too.")
    p.add_argument('--perc-sigma', type=float, default=0.6,
                   help='only apply it below this sigma, where x0 is close enough to be worth decoding')
    p.add_argument('--perc-crop', type=int, default=16, help='latent crop side; 16 -> 256x256 pixels')
    p.add_argument('--guide-key', default='guide',""")

old = """    def forward(target, guide, sigma, gen):"""
new = """    vae = None
    lpips_net = None
    if a.perc > 0:
        import lpips as _lp
        sd_vae, meta_vae = comfy.utils.load_torch_file(str(VAE_PATH), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd_vae, device=dev, dtype=torch.float16, metadata=meta_vae)
        vae.first_stage_model.eval().requires_grad_(False)
        del sd_vae
        lpips_net = _lp.LPIPS(net='alex', version='0.1', verbose=False).eval().to(dev)
        for q in lpips_net.parameters():
            q.requires_grad_(False)

    def perceptual(x0_hat, target, gen):
        \"\"\"LPIPS between decoded x0 and the decoded target, on a random latent crop.

        Two latent frames is the smallest unit the H3 VAE accepts (its temporal layout is 2+5k, so 2 gives
        5 pixel frames), and a 16x16 latent crop keeps the decode that the gradient has to pass through
        small enough to fit alongside the DiT's own activations.\"\"\"
        c = a.perc_crop
        H16, W16 = x0_hat.shape[-2], x0_hat.shape[-1]
        y0 = int(torch.randint(0, max(1, H16 - c + 1), (1,), generator=gen, device=dev).item())
        x0 = int(torch.randint(0, max(1, W16 - c + 1), (1,), generator=gen, device=dev).item())
        sl = (slice(None), slice(None), slice(0, 2), slice(y0, y0 + c), slice(x0, x0 + c))
        dec_hat = vae.decode(x0_hat[sl].to(torch.float16))
        with torch.no_grad():
            dec_tgt = vae.decode(target[sl].to(torch.float16))
        f = lambda z: (z[0] if z.ndim == 5 else z).permute(0, 3, 1, 2).float().clamp(0, 1) * 2 - 1
        return lpips_net(f(dec_hat), f(dec_tgt)).mean()

    def forward(target, guide, sigma, gen):"""
assert old in s
s = s.replace(old, new, 1)

old2 = """        return F.mse_loss(out[0].float(), eps - target)"""
new2 = """        v = out[0].float()
        mse = F.mse_loss(v, eps - target)
        if a.perc <= 0 or sigma > a.perc_sigma:
            return mse
        # flow matching: xt = x + s*(eps - x), so x0 = xt - s*v
        x0_hat = (xt.float() - s * v)
        return mse + a.perc * perceptual(x0_hat, target.float(), gen)"""
assert old2 in s
s = s.replace(old2, new2)

s = s.replace("from h3_comfy import", "from h3_comfy import VAE_PATH,") if 'VAE_PATH' not in s.split('def main')[0] else s
p.write_text(s)
import ast; ast.parse(s)
print('patched perceptual loss')
