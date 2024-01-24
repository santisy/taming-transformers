import argparse
import glob
import yaml
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
from rich.console import Console
import tqdm
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics import StructuralSimilarityIndexMeasure
import lpips

console = Console()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.* x - 1.
  return x


def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  xrec = model.decode(z)
  return xrec

def open_image(path):
    return PIL.Image.open(path)


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    #if map_dalle: 
    #  img = map_pixels(img)
    return img


def stack_reconstructions_my(input, x2):
  assert input.size == x2.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (w, h))
  img.paste(x2, (0,0))
  return img


def reconstruction_pipeline(model, path, size=256):
  # x_dalle = preprocess(open_image(path), target_image_size=size, map_dalle=True)
  x_vqgan = preprocess(open_image(path), target_image_size=size, map_dalle=False)
  x_vqgan = x_vqgan.to(DEVICE)
  x_vqgan = preprocess_vqgan(x_vqgan)
  x2, _ = model(x_vqgan)#reconstruct_with_vqgan(x_vqgan, model)
  #print(x2.shape)
  #x2 = Image.fromarray(x2.detach().cpu().numpy())
  img = custom_to_pil(x2[0])
  return img, x2, x_vqgan



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--out_dir', required=True, type=str)
    parser.add_argument('--max_save', type=int, default=300)

    args = parser.parse_args()

    configs = load_config(glob.glob(os.path.join(args.log_dir, 'configs',
                                                '*-project.yaml'))[0],
                        display=False)
    console.print('Loaded configs.', style="green")
    model = load_vqgan(configs,
                       ckpt_path=os.path.join(args.log_dir,
                                              'checkpoints',
                                              'last.ckpt')).to(DEVICE).eval()
    console.print('Constructed and loaded the model.', style="green")
    data_dir = args.data_dir
    out_dir = args.out_dir

    # Preparation
    os.makedirs(out_dir, exist_ok=True)
    file_names = glob.glob(os.path.join(data_dir, '*.png'))
    data_len = len(file_names)
    f = open(os.path.join(out_dir, 'metrics.txt'), 'a')
    pbar = tqdm.tqdm(total=data_len)

    # Initiate metrics

    loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
    psnr_fn = PeakSignalNoiseRatio().to(DEVICE)
    ssim_fn = StructuralSimilarityIndexMeasure().to(DEVICE)

    lpips_collect = torch.zeros(data_len).to(DEVICE)
    psnr_collect = torch.zeros(data_len).to(DEVICE)
    ssim_collect = torch.zeros(data_len).to(DEVICE)

    console.print('Initialized metrics.', style="green")
    console.print('Begin running loop.', style="green")

    for i, f_path in enumerate(file_names):
        f_name = os.path.basename(f_path)
        with torch.no_grad():
          img, output_tensor, input_tensor = reconstruction_pipeline(
            model, path=f'{data_dir}/{f_name}', size=256)
        if args.max_save > 0 and i < args.max_save:
            img.save(f'{out_dir}/{f_name}')

        lpips_collect[i] = loss_fn(input_tensor, output_tensor).detach()
        psnr_collect[i] = psnr_fn(output_tensor, input_tensor).detach()
        ssim_collect[i] = ssim_fn(output_tensor, input_tensor).detach()
        pbar.update(1)


    # Record LPIPS
    lpips_collect = lpips_collect.mean(dim=0)
    f.write(f'LPIPS: \nmean {lpips_collect.mean().item(): .4f} \t'
            f' 2*std {lpips_collect.std().item() * 2: .6f}\n')
    # Record PSNR
    psnr_collect = psnr_collect.mean(dim=0)
    f.write(f'PSNR: \nmean {psnr_collect.mean().item(): .4f} \t'
            f' 2*std {psnr_collect.std().item() * 2: .6f}\n')
    # Record SSIM
    ssim_collect = ssim_collect.mean(dim=0)
    f.write(f'SSIM: \nmean {ssim_collect.mean().item(): .4f} \t'
            f' 2*std {ssim_collect.std().item() * 2: .6f}\n')

    f.close()
    pbar.close()
