import argparse
import math
import os

import torch
import torchvision
from torch import optim
from tqdm import tqdm

from criteria.clip_loss import CLIPLoss
from mapper.training.train_utils import STYLESPACE_DIMENSIONS
from models.stylegan2.model import Generator
import clip

STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in list(range(1, len(STYLESPACE_DIMENSIONS), 3))]

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)

    latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    with torch.no_grad():
        img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)

    if args.work_in_stylespace:
        with torch.no_grad():
            _, _, latent_code_init = g_ema([latent_code_init], input_is_latent=True, return_latents=True)
        latent = [s.detach().clone() for s in latent_code_init]
        for c, s in enumerate(latent):
            if c in STYLESPACE_INDICES_WITHOUT_TORGB:
                s.requires_grad = True
    else:
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True

    clip_loss = CLIPLoss(args)

    if args.work_in_stylespace:
        optimizer = optim.Adam(latent, lr=args.lr)
    else:
        optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

        c_loss = clip_loss(img_gen, text_inputs)

        loss = c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False, input_is_stylespace=args.work_in_stylespace)

            torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.jpg", normalize=True, range=(-1, 1))

    final_result = img_gen

    return final_result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="a person with purple hair", help="the text that guides the generation")
    parser.add_argument("--ckpt", type=str, default="./013500.pt", help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=256, help="StyleGAN resolution")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=300, help="number of optimization steps")
    parser.add_argument('--work_in_stylespace', default=False, action='store_true')
    parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"), normalize=True, scale_each=True, range=(-1, 1))


