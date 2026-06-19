import os
import sys

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import random

from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from skimage import morphology
from skimage.transform import resize

from scipy.ndimage import binary_dilation, gaussian_filter, distance_transform_edt

from Trainer.models import LDM_3D
from datasets import USBData
from utils.get_betas import get_betas
from utils.denoise import (
    denoise_uncond,
    denoise_cond,
    denoise_p2h,
    denoise_h2p,
)

from autoencoders import AutoencoderKLAD

def load_nii_as_tensor(path, target_shape=(160,160,160)):

        nii = nib.load(path)
        img = nii.get_fdata().astype(np.float32)

        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

        # img = F.interpolate(img, size=target_shape, mode='trilinear', align_corners=False)
        return img.squeeze(0)

def get_slice(slice):

    target_size = (230, 182)
    slice = resize(slice, target_size, order=1, preserve_range=True, anti_aliasing=True)

    h, w = slice.shape
    max_side = max(h, w)
    pad_top = (max_side - h) // 2
    pad_bottom = max_side - h - pad_top
    pad_left = (max_side - w) // 2
    pad_right = max_side - w - pad_left

    slice = np.pad(slice, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    return slice

'''
def save_volume_and_slice(
        volume,
        result_dir,
        name_prefix,
        slice_idx=60,
        do_binary=False,
        affine=None
    ):

    if affine is None:
        affine = np.eye(4)

    vol_path = os.path.join(result_dir, f"{name_prefix}.nii.gz")

    nib.save(nib.Nifti1Image(volume, affine=affine), vol_path)

    mid_slice = np.rot90(volume[:, slice_idx, :])

    if do_binary:
        mid_slice = (mid_slice - volume.min()) / (volume.max() - volume.min() + 1e-8)
        mid_slice = (mid_slice > 0.5)
        mid_slice = morphology.remove_small_holes(mid_slice, area_threshold=50).astype(np.float32)

    mid_slice = get_slice(mid_slice)

    slice_path = os.path.join(result_dir, f"{name_prefix}.png")
    

    plt.imsave(slice_path, mid_slice, cmap="gray")

    return vol_path, slice_path
'''

def normalize_volume_for_save(volume: np.ndarray, clip_percentiles=(0.5, 99.5)) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=np.float32)


    lo, hi = np.percentile(nonzero, clip_percentiles)
    volume = np.clip(volume, lo, hi)


    vmin = float(volume.min())
    vmax = float(volume.max())


    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)


    return volume.astype(np.float32)




def save_volume_and_slice(
        volume,
        result_dir,
        name_prefix,
        slice_idx=60,
        do_binary=False,
        affine=None
    ):


    if affine is None:
        affine = np.eye(4, dtype=np.float32)


    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


    if do_binary:
        volume_to_save = (volume > 0.5).astype(np.float32)
    else:
        #volume_to_save = normalize_volume_for_save(volume)
        volume_to_save = np.clip(volume, 0.0, 1.0).astype(np.float32)


    vol_path = os.path.join(result_dir, f"{name_prefix}.nii.gz")
    nib.save(nib.Nifti1Image(volume_to_save, affine=affine), vol_path)


    slice_idx = max(0, min(slice_idx, volume_to_save.shape[1] - 1))
    mid_slice = np.rot90(volume_to_save[:, slice_idx, :])


    if do_binary:
        mid_slice = (mid_slice > 0.5).astype(np.float32)
        mid_slice = morphology.remove_small_holes(mid_slice, area_threshold=50).astype(np.float32)


    mid_slice = get_slice(mid_slice)


    slice_path = os.path.join(result_dir, f"{name_prefix}.png")
    plt.imsave(slice_path, mid_slice, cmap="gray", vmin=0.0, vmax=1.0)


    return vol_path, slice_path


class Test():
    def __init__(self, args):
        # load configs
        self.model_config = args.model
        self.diffusion_config = args.diffusion
        self.data_config = args.data

        if args.mode == "uncond_gen":
            self.sample_config = args.uncond_gen

        elif args.mode == "cond_gen":
            self.sample_config = args.cond_gen
        elif args.mode == "p2h_edit":
            self.edit_config = args.p2h_edit
        elif args.mode == "h2p_edit":
            self.edit_config = args.h2p_edit
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.betas = get_betas(self.diffusion_config).to(self.device)
        self.num_diffusion_timesteps = self.diffusion_config.num_diffusion_timesteps

        # load VAE
        if not os.path.exists(self.model_config.vae_path):
            raise ValueError(f"Model not found at {self.model_config.vae_path}")
        self.vae = AutoencoderKLAD.load_from_checkpoint(self.model_config.vae_path, cfg=self.model_config.vae_config_path, input_dim=[(160, 160, 160)])
        self.vae.eval() 
        self.vae.requires_grad_(False)
        
    def uncond_gen(self):
        """
        Unconditional Generation
        """

        model_config = self.model_config
        sample_config = self.sample_config
        device = self.device
        img_num = sample_config.img_num
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = sample_config.eta
        assert sample_config.skip_type == 'uniform' or sample_config.skip_type == 'quad', print(
            sample_config.skip_type)
        
        if sample_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // sample_config.num_sample_timesteps
            denoise_t_list = range(0, num_diffusion_timesteps, skip)
        elif sample_config.multi_sample.skip_type == "quad":
            denoise_t_list = (np.linspace(0, np.sqrt(num_diffusion_timesteps * 0.8),
                                          sample_config.num_sample_timesteps) ** 2)
            denoise_t_list = [int(s) for s in list(denoise_t_list)]
        else:
            raise NotImplementedError("Skip type not defined")
        
        x_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels, 
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)
        x_model_path = model_config.x_model_path
        x_states = torch.load(x_model_path)
        x_model = x_model.to(device)
        x_model = torch.nn.DataParallel(x_model)
        x_model.load_state_dict(x_states[0], strict=True)


        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)
        y_model_path = model_config.y_model_path
        y_states = torch.load(y_model_path)
        y_model = y_model.to(device)
        y_model = torch.nn.DataParallel(y_model)
        y_model.load_state_dict(y_states[0], strict=True)
        x_model.eval()
        y_model.eval()

        with torch.no_grad():
            batch_size = sample_config.batch_size
            batch_num = np.ceil(img_num / batch_size).astype(int)
            result_dir = sample_config.result_dir
            for batch in range(batch_num):
                n_batch = min(batch_size, img_num - batch * batch_size)

                xt = torch.randn((n_batch, model_config.in_channels) + tuple(model_config.initial_resolution)).to(device)
                yt = torch.randn((n_batch, model_config.in_channels) + tuple(model_config.initial_resolution)).to(device)
                xts, yts, x0s, y0s = denoise_uncond(x_model, y_model, xt, yt, denoise_t_list, betas, eta)

                x0, y0 = x0s[-1], y0s[-1]
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                self.vae = self.vae.to(device)

                x0 = 1 / 0.18215 * x0
                x0 = self.vae.decode([x0])[0]._sample()

                y0 = 1 / 0.18215 * y0
                y0 = self.vae.decode([y0])[0]._sample()
                
                x0 = x0.detach().cpu().numpy()
                y0 = y0.detach().cpu().numpy()                   
                
                for i in tqdm(range(x0.shape[0]), desc='saving'):

                    volume_x = x0[i, 0]
                    volume_y = y0[i, 0]

                    idx = batch * batch_size + i

                    save_volume_and_slice(volume_x, result_dir, f"x0_{idx}", do_binary=True)
                    save_volume_and_slice(volume_y, result_dir, f"y0_{idx}", do_binary=False)
    

    def cond_gen(self):
        """
        Conditional Generation
        """

        model_config = self.model_config
        sample_config = self.sample_config
        device = self.device
        img_num = sample_config.img_num
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = sample_config.eta

        assert sample_config.skip_type in ['uniform', 'quad'], print(sample_config.skip_type)
        
        if sample_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // sample_config.num_sample_timesteps
            denoise_t_list = range(0, num_diffusion_timesteps, skip)
        elif sample_config.skip_type == 'quad':
            denoise_t_list = (np.linspace(0, np.sqrt(num_diffusion_timesteps * 0.8),
                                        sample_config.num_sample_timesteps) ** 2)
            denoise_t_list = [int(s) for s in list(denoise_t_list)]
        else:
            raise NotImplementedError("Skip type not defined")

        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)

        y_states = torch.load(model_config.y_model_path)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        
        y_model.eval()

        x_file_list = sample_config.x_data_path
        with open(sample_config.x_data_path, 'r') as f: 
            x_files = [line.strip() for line in f.readlines()]

        x_files = sorted(x_files)
        print(f"Found {len(x_files)} NIfTI files in {x_file_list}")

        with torch.no_grad():
            result_dir = sample_config.result_dir
            os.makedirs(result_dir, exist_ok=True)

            for idx, nii_file in enumerate(tqdm(x_files, desc="Generating")):
                x0 = load_nii_as_tensor(nii_file).to(device)
                x0 = x0.unsqueeze(0)

                self.vae = self.vae.to(device)
                x0 = self.vae.encode([x0])[0]._sample().mul_(0.18215) 
                x0 = x0.to(device)

                yt = torch.randn_like(x0).to(device)

                yts, y0s = denoise_cond(y_model, x0, yt, denoise_t_list, betas, eta)

                x0 = 1 / 0.18215 * x0
                x0 = self.vae.decode([x0])[0]._sample()

                y0 = y0s[-1]
                y0 = 1 / 0.18215 * y0
                y0 = self.vae.decode([y0])[0]._sample()

                x0 = x0.detach().cpu().numpy().astype(np.float32)
                y0 = y0.detach().cpu().numpy().astype(np.float32)

                volume_x = x0[0, 0]
                volume_y = y0[0, 0]
                
                case_name = os.path.splitext(os.path.basename(nii_file))[0]

                save_volume_and_slice(volume_x, result_dir, f"x0_{case_name}_{idx}", do_binary=True)
                save_volume_and_slice(volume_y, result_dir, f"y0_{case_name}_{idx}", do_binary=False)
                        
    def p2h_edit(self):
        """
        Pathology-to-Healthy Editing
        """
        #data_config = self.data_config
        model_config = self.model_config
        edit_config = self.edit_config
        #data_config.data_file = edit_config.y_data_path

        device = self.device
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = edit_config.eta

        result_dir = edit_config.result_dir
        os.makedirs(result_dir, exist_ok=True)

        assert edit_config.skip_type in ['uniform', 'quad'], f"Unknown skip_type: {edit_config.skip_type}"
        
        if edit_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        elif edit_config.skip_type == 'quad':
            skip = max(1, int(np.sqrt(num_diffusion_timesteps) // np.sqrt(edit_config.num_sample_timesteps)))

        #dataset = USBData(data_config, training_=False, device=device)
        #edit_dataloader = data.DataLoader(
        #    dataset, batch_size=edit_config.batch_size, shuffle=False, num_workers=edit_config.num_workers
        #)

        def strip_nii_ext(path: str) -> str:
            base = os.path.basename(path)
            if base.endswith(".nii.gz"):
                return base[:-7]
            if base.endswith(".nii"):
                return base[:-4]
            return os.path.splitext(base)[0]


        def load_txt_paths(txt_path: str) -> list[str]:
            with open(txt_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines() if line.strip()]


        def load_image_tensor(path: str) -> torch.Tensor:
            arr = nib.load(path).get_fdata().astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


            arr_min = float(arr.min())
            arr_max = float(arr.max())


            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)


            arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(0)

        def load_mask_tensor(path: str, dilation_iters: int = 2) -> torch.Tensor:
            seg = nib.load(path).get_fdata().astype(np.float32)
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

            # Full affected region.
            # BraTS-style: include labels 1, 2, 4.
            if np.max(seg) > 1.0:
                mask = seg > 0
            else:
                mask = seg > 0.5

            if dilation_iters > 0:
                mask = binary_dilation(mask, iterations=dilation_iters)

            return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        def make_prefilled_image(
            img: torch.Tensor,
            mask: torch.Tensor,
            shell_iters: int = 8,
            smooth_sigma: float = 0.8,
            fill_percentile: float = 75.0,
            texture_strength: float = 0.05,
            mirror_axis: int = 0,
            mirror_weight: float = 0.85,
        ) -> torch.Tensor:
            """
            img:  [B,1,H,W,D], normalized [0,1]
            mask: [B,1,H,W,D], binary/soft mask


            Contralateral pseudo-healthy prefill:
            - inside lesion: mostly mirrored tissue from opposite hemisphere
            - fallback: local bright shell fill
            - boundary smoothing only inside mask
            """
            img_np = img.detach().cpu().numpy().astype(np.float32)
            mask_np = mask.detach().cpu().numpy().astype(np.float32)


            out = img_np.copy()


            for b in range(img_np.shape[0]):
                vol = img_np[b, 0]
                m = mask_np[b, 0] > 0.5

                brain = vol > 0.05

                local_outer = binary_dilation(m, iterations=shell_iters)
                local_shell = local_outer & (~m) & brain

                if np.any(local_shell):
                    shell_vals = vol[local_shell]
                else:
                    shell_vals = vol[brain]

                white_target = float(np.percentile(shell_vals, 80))
                dark_core = m & (vol < np.percentile(shell_vals, 35))

                if not np.any(m):
                    continue


                brain = vol > 0.05


                # Local shell statistics
                shell_outer = binary_dilation(m, iterations=shell_iters)
                shell_inner = binary_dilation(m, iterations=2)
                shell = shell_outer & (~shell_inner) & brain


                if np.any(shell):
                    local_vals = vol[shell]
                elif np.any(brain):
                    local_vals = vol[brain]
                else:
                    local_vals = vol.reshape(-1)


                local_fill = float(np.percentile(local_vals, fill_percentile))
                local_std = float(np.std(local_vals))


                # Mirror tissue from opposite hemisphere
                mirrored = np.flip(vol, axis=mirror_axis)


                # Basic local intensity correction:
                # match mirrored tissue median to local shell median
                mirrored_vals = mirrored[shell] if np.any(shell) else mirrored[brain]
                if mirrored_vals.size > 0:
                    mirrored_median = float(np.median(mirrored_vals))
                    local_median = float(np.median(local_vals))
                    mirrored_corrected = mirrored + (local_median - mirrored_median)
                else:
                    mirrored_corrected = mirrored.copy()


                mirrored_corrected = np.clip(mirrored_corrected, 0.0, 1.0)


                # Small texture fallback, not huge noise
                noise = np.random.normal(0.0, 1.0, vol.shape).astype(np.float32)
                texture = gaussian_filter(noise, sigma=2.0)
                texture = (texture - texture.mean()) / (texture.std() + 1e-8)


                local_synthetic = local_fill + texture_strength * local_std * texture
                local_synthetic = np.clip(
                    local_synthetic,
                    np.percentile(local_vals, 20),
                    np.percentile(local_vals, 98),
                )


                # Mostly mirror prior, small local-statistical fallback
                prefill = (
                    mirror_weight * mirrored_corrected
                    + (1.0 - mirror_weight) * local_synthetic
                )

                # Aggressively remove necrotic/dark cores
                core_target = np.maximum(prefill, white_target)

                prefill[dark_core] = (
                    0.20 * prefill[dark_core]
                    + 0.80 * core_target[dark_core]
                )

                filled = vol.copy()
                filled[m] = prefill[m]


                # Smooth only the inserted region
                smoothed = gaussian_filter(filled, sigma=smooth_sigma)
                filled[m] = 0.85 * filled[m] + 0.15 * smoothed[m]


                out[b, 0] = np.clip(filled, 0.0, 1.0)


            return torch.from_numpy(out).to(img.device)




        y_p_paths = load_txt_paths(edit_config.y_data_path)
        mask_paths = load_txt_paths(edit_config.x_data_path)


        if len(y_p_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch: {len(y_p_paths)} pathological images vs {len(mask_paths)} masks"
            )


        pairs = list(zip(y_p_paths, mask_paths))

        batch_size = edit_config.batch_size
        
        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)

        y_model_path = model_config.y_model_path
        y_states = torch.load(y_model_path, map_location=device)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        y_model.eval()


        t_start = getattr(edit_config, "t_start", 100)
        skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        denoise_t_list = range(0, t_start, skip)

        with torch.no_grad():
            #with tqdm(total=len(edit_dataloader), desc=f'Editing', file=sys.__stdout__) as pbar:
                #for i, batch in enumerate(edit_dataloader):
            with tqdm(total=len(pairs), desc="Editing", file=sys.__stdout__) as pbar:
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start:start + batch_size]

                    #y_p = batch[edit_config.y].to(device)

                    '''y_p_list = []
                    img_names = []
                    affines = []

                    for path in batch_paths:
                        nii = nib.load(path)
                        affines.append(nii.affine)
                        y_p_list.append(load_image_tensor(path))
                        img_names.append(strip_nii_ext(path))'''

                    y_p_list = []
                    mask_list = []
                    img_names = []
                    affines = []

                    mask_dilation = getattr(edit_config, "mask_dilation", 2)

                    for img_path, mask_path in batch_pairs:
                        nii = nib.load(img_path)
                        affines.append(nii.affine)

                        y_p_list.append(load_image_tensor(img_path))
                        mask_list.append(load_mask_tensor(mask_path, dilation_iters=mask_dilation))

                        img_names.append(strip_nii_ext(img_path))

                    y_p_img = torch.stack(y_p_list, dim=0).to(device)
                    mask_full = torch.stack(mask_list, dim=0).to(device)

                    #y_p = torch.stack(y_p_list, dim=0).to(device)

                    #self.vae = self.vae.to(device)

                    '''y_p = self.vae.encode([y_p])[0]._sample().mul_(0.18215).to(device)
                    
                    t = torch.full((y_p.shape[0],), t_start, device=device, dtype=torch.long)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
                    e_y = torch.randn_like(y_p)
                    yt = y_p * a.sqrt() + e_y * (1.0 - a).sqrt()

                    xt = torch.zeros_like(y_p).to(device)

                    xts, yts, x0s, y0s = denoise_p2h(
                        y_model, xt, yt, y_p, denoise_t_list, betas, eta,
                        alpha0=20.0, 
                        decay=0.5,
                    )

                    y0 = y0s[-1]
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    self.vae = self.vae.to(device)

                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()

                    y_p = 1 / 0.18215 * y_p
                    y_p = self.vae.decode([y_p])[0]._sample()
                    
                    y0 = y0.detach().cpu().numpy()
                    y_p = y_p.detach().cpu().numpy()'''

                    '''
                    for i in tqdm(range(y0.shape[0]), desc='saving'):
                        volume_y_h = y0[i, 0]
                        volume_y_p = y_p[i, 0]

                        case_name = batch['img_name'][i] + '_' + batch['pathol_name'][i]
                        save_volume_and_slice(volume_y_h, result_dir, f"y_h_{case_name}", do_binary=False)
                        save_volume_and_slice(volume_y_p, result_dir, f"y_p_{case_name}", do_binary=False)
                    '''

                    #y_p = self.vae.encode([y_p])[0]._sample().mul_(0.18215).to(device)

                    self.vae = self.vae.to(device)

                    '''# Keep original image-space pathological image for final hard blending
                    y_p_original_img = y_p_img.clone()

                    # Encode pathological image
                    y_p = self.vae.encode([y_p_img])[0]._sample().mul_(0.18215).to(device)'''

                    y_p_original_img = y_p_img.clone()

                    prefill_shell_iters = getattr(edit_config, "prefill_shell_iters", 6)
                    prefill_smooth_sigma = getattr(edit_config, "prefill_smooth_sigma", 1.0)
                    prefill_fill_percentile = getattr(edit_config, "prefill_fill_percentile", 75.0)
                    prefill_texture_strength = getattr(edit_config, "prefill_texture_strength", 0.15)
                    prefill_mirror_axis = getattr(edit_config, "prefill_mirror_axis", 0)
                    prefill_mirror_weight = getattr(edit_config, "prefill_mirror_weight", 0.85)

                    y_p_prefilled_img = make_prefilled_image(
                        y_p_img,
                        mask_full,
                        shell_iters=prefill_shell_iters,
                        smooth_sigma=prefill_smooth_sigma,
                        fill_percentile=prefill_fill_percentile,
                        texture_strength=prefill_texture_strength,
                        mirror_axis=prefill_mirror_axis,
                        mirror_weight=prefill_mirror_weight,
                    )

                    # Encode the crude pseudo-healthy image, not the raw lesioned image.
                    y_p = self.vae.encode([y_p_prefilled_img])[0]._sample().mul_(0.18215).to(device)

                    # Convert full-res mask [B,1,160,160,160] to latent [B,1,20,20,20]
                    # IMPORTANT: max-pooling prevents small lesions from disappearing.
                    scale = mask_full.shape[-1] // y_p.shape[-1]  # usually 160 // 20 = 8

                    x0_mask_lcg = F.max_pool3d(
                        mask_full,
                        kernel_size=scale,
                        stride=scale,
                    )

                    x0_mask_lcg = (x0_mask_lcg > 0.0).float()

                    latent_dilation = getattr(edit_config, "latent_dilation", 1)
                    if latent_dilation > 0:
                        for _ in range(latent_dilation):
                            x0_mask_lcg = F.max_pool3d(
                                x0_mask_lcg,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                            )

                    x0_mask_lcg = torch.clamp(x0_mask_lcg, 0.0, 1.0)

                    t_start = getattr(edit_config, "t_start", 120)
                    alpha0 = getattr(edit_config, "alpha0", 15.0)
                    decay = getattr(edit_config, "decay", 0.3)

                    skip = max(1, num_diffusion_timesteps // edit_config.num_sample_timesteps)
                    denoise_t_list = range(0, t_start, skip)

                    t = torch.full((y_p.shape[0],), t_start, device=device, dtype=torch.long)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)

                    e_y = torch.randn_like(y_p)

                    yt = y_p * a.sqrt() + e_y * (1.0 - a).sqrt()
                    #mask_latent = x0_mask_lcg
                    #if mask_latent.shape[1] == 1:
                    #    mask_latent = mask_latent.repeat(1, y_p.shape[1], 1, 1, 1)

                    #yt_normal = y_p * a.sqrt() + e_y * (1.0 - a).sqrt()
                    #yt_noise_inside = e_y

                    #yt = (1.0 - mask_latent) * yt_normal + mask_latent * yt_noise_inside

                    # Healthy condition: no lesion mask
                    xt = torch.zeros_like(y_p).to(device)

                    xts, yts, x0s, y0s = denoise_p2h(
                        y_model,
                        xt,
                        yt,
                        y_p,
                        denoise_t_list,
                        betas,
                        eta,
                        alpha0=alpha0,
                        decay=decay,
                        x0_mask_lcg=x0_mask_lcg,
                        preserve_outside_mask=True,
                    )

                    y0 = y0s[-1]

                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()

                    y0_np = y0.detach().cpu().numpy().astype(np.float32)
                    y_p_original_np = y_p_original_img.detach().cpu().numpy().astype(np.float32)
                    mask_np = mask_full.detach().cpu().numpy().astype(np.float32)

                    for i in range(y0_np.shape[0]):
                        volume_y_h_raw = y0_np[i, 0]
                        volume_y_p_raw = y_p_original_np[i, 0]
                        mask_i = mask_np[i, 0]

                        mask_hard = mask_i > 0.5
                        mask_soft = gaussian_filter(mask_hard.astype(np.float32), sigma=0.7)
                        mask_soft = np.clip(mask_soft, 0.0, 1.0)

                        # Force full replacement in the core
                        mask_core = binary_dilation(mask_hard, iterations=1)
                        mask_soft[mask_core] = 1.0

                        volume_y_h = (
                            (1.0 - mask_soft) * volume_y_p_raw
                            + mask_soft * volume_y_h_raw
                        ).astype(np.float32)

                        ''' # Soft boundary for less visible seams
                        mask_soft = gaussian_filter(mask_i.astype(np.float32), sigma=1.0)
                        mask_soft = np.clip(mask_soft, 0.0, 1.0)

                        # Absolute protection: outside mask comes from original pathological input
                        volume_y_h = (
                            (1.0 - mask_soft) * volume_y_p_raw
                            + mask_soft * volume_y_h_raw
                        ).astype(np.float32)'''

                        case_name = img_names[i]

                        save_volume_and_slice(
                            volume_y_h,
                            result_dir,
                            f"y_h_{case_name}",
                            do_binary=False,
                            affine=affines[i],
                        )

                        save_volume_and_slice(
                            volume_y_p_raw,
                            result_dir,
                            f"y_p_{case_name}",
                            do_binary=False,
                            affine=affines[i],
                        )

                    pbar.update(len(batch_pairs))


                    '''xt = torch.zeros_like(y_p).to(device)

                    # -------------------------------------------------
                    # PASS 1: weak p2h reconstruction
                    # Purpose: estimate where USB wants to change tissue.
                    # -------------------------------------------------
                    t_start_pass1 = getattr(edit_config, "t_start_pass1", 100)
                    alpha0_pass1 = getattr(edit_config, "alpha0_pass1", 20.0)
                    decay_pass1 = getattr(edit_config, "decay_pass1", 0.5)


                    t1 = torch.full((y_p.shape[0],), t_start_pass1, device=device, dtype=torch.long)
                    a1 = (1 - betas).cumprod(dim=0).index_select(0, t1).view(-1, 1, 1, 1, 1)
                    e1 = torch.randn_like(y_p)
                    yt1 = y_p * a1.sqrt() + e1 * (1.0 - a1).sqrt()


                    denoise_t_list_pass1 = range(
                        0,
                        t_start_pass1,
                        max(1, num_diffusion_timesteps // edit_config.num_sample_timesteps),
                    )


                    _, _, _, y0s_pass1 = denoise_p2h(
                        y_model,
                        xt,
                        yt1,
                        y_p,
                        denoise_t_list_pass1,
                        betas,
                        eta,
                        alpha0=alpha0_pass1,
                        decay=decay_pass1,
                        x0_mask_lcg=None,
                        preserve_outside_mask=False,
                    )


                    y_weak = y0s_pass1[-1]


                    # -------------------------------------------------
                    # PSEUDO-MASK: latent anomaly estimate
                    # Based on reconstruction difference.
                    # -------------------------------------------------
                    diff = torch.mean(torch.abs(y_p - y_weak), dim=1, keepdim=True)


                    brain = torch.mean(torch.abs(y_p), dim=1, keepdim=True)
                    brain = brain > torch.quantile(brain.flatten(), 0.20)


                    diff_masked = diff[brain]
                    if diff_masked.numel() > 0:
                        q = getattr(edit_config, "pseudo_mask_quantile", 0.95)
                        threshold = torch.quantile(diff_masked, q)
                    else:
                        threshold = torch.quantile(diff.flatten(), 0.95)


                    pseudo_mask = (diff >= threshold).float()
                    pseudo_mask = pseudo_mask * brain.float()


                    # clean / slightly expand in latent space
                    pseudo_mask = F.max_pool3d(
                        pseudo_mask,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )


                    pseudo_mask = F.avg_pool3d(
                        pseudo_mask,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )


                    pseudo_mask = (pseudo_mask > 0.15).float()


                    # -------------------------------------------------
                    # PASS 2: constrained p2h
                    # Purpose: only edit estimated anomaly region.
                    # -------------------------------------------------
                    t_start_pass2 = getattr(edit_config, "t_start_pass2", getattr(edit_config, "t_start", 150))
                    alpha0_pass2 = getattr(edit_config, "alpha0_pass2", 15.0)
                    decay_pass2 = getattr(edit_config, "decay_pass2", 0.3)


                    t2 = torch.full((y_p.shape[0],), t_start_pass2, device=device, dtype=torch.long)
                    a2 = (1 - betas).cumprod(dim=0).index_select(0, t2).view(-1, 1, 1, 1, 1)
                    e2 = torch.randn_like(y_p)
                    yt2 = y_p * a2.sqrt() + e2 * (1.0 - a2).sqrt()


                    denoise_t_list_pass2 = range(
                        0,
                        t_start_pass2,
                        max(1, num_diffusion_timesteps // edit_config.num_sample_timesteps),
                    )


                    xts, yts, x0s, y0s = denoise_p2h(
                        y_model,
                        xt,
                        yt2,
                        y_p,
                        denoise_t_list_pass2,
                        betas,
                        eta,
                        alpha0=alpha0_pass2,
                        decay=decay_pass2,
                        x0_mask_lcg=pseudo_mask,
                        preserve_outside_mask=True,
                    )


                    y0 = y0s[-1]


                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)


                    self.vae = self.vae.to(device)


                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()


                    y_p = 1 / 0.18215 * y_p
                    y_p = self.vae.decode([y_p])[0]._sample()


                    y0 = y0.detach().cpu().numpy()
                    y_p = y_p.detach().cpu().numpy()

                    for i in range(y0.shape[0]):
                        volume_y_h = y0[i, 0]
                        volume_y_p = y_p[i, 0]


                        case_name = img_names[i]


                        save_volume_and_slice(
                            volume_y_h,
                            result_dir,
                            f"y_h_{case_name}",
                            do_binary=False,
                            affine=affines[i],
                        )


                        save_volume_and_slice(
                            volume_y_p,
                            result_dir,
                            f"y_p_{case_name}",
                            do_binary=False,
                            affine=affines[i],
                        )


                    pbar.update(len(batch_paths))'''

    '''
    def h2p_edit(self):
        """
        Healthy-to-Pathology Editing
        """
        data_config = self.data_config
        model_config = self.model_config
        edit_config = self.edit_config
        data_config.data_file = edit_config.y_data_path

        device = self.device
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = edit_config.eta

        result_dir = edit_config.result_dir

        assert edit_config.skip_type in ['uniform', 'quad'], f"Unknown skip_type: {edit_config.skip_type}"
        
        if edit_config.skip_type == 'uniform':
            skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        elif edit_config.skip_type == 'quad':
            skip = max(1, int(np.sqrt(num_diffusion_timesteps) // np.sqrt(edit_config.num_sample_timesteps)))

        dataset = USBData(data_config, training_=False, device=device)
        edit_dataloader = data.DataLoader(
            dataset, batch_size=edit_config.batch_size, shuffle=False, num_workers=edit_config.num_workers
        )

        y_model = LDM_3D(in_channels=model_config.in_channels,
                        out_channels=model_config.in_channels,
                        num_channels=model_config.num_channels,
                        attention_levels=model_config.attention_levels,
                        num_res_blocks=model_config.num_res_blocks,
                        num_head_channels=model_config.num_head_channels,
                        with_conditioning=False,
                        learn_sigma=model_config.learn_sigma)

        y_model_path = model_config.y_model_path
        y_states = torch.load(y_model_path, map_location=device)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        y_model.eval()

        t_start = getattr(edit_config, "t_start", 500)
        skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        denoise_t_list = range(0, t_start, skip)

        with torch.no_grad():
            with tqdm(total=len(edit_dataloader), desc=f'Editing', file=sys.__stdout__) as pbar:
                for i, batch in enumerate(edit_dataloader):

                    y_h = batch['img_wo_pathol'].to(device)

                    x0_ori = batch[edit_config.x].to(device)

                    self.vae = self.vae.to(device)

                    y_h = self.vae.encode([y_h])[0]._sample().mul_(0.18215).to(device)
                    x0_ori = self.vae.encode([x0_ori])[0]._sample().mul_(0.18215).to(device)

                    t = torch.full((y_h.shape[0],), t_start, device=device, dtype=torch.long)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)
                    e_y = torch.randn_like(y_h)
                    yt = y_h * a.sqrt() + e_y * (1.0 - a).sqrt()
                    xt = x0_ori

                    xts, yts, x0s, y0s = denoise_h2p(
                        y_model, xt, yt, y_h, denoise_t_list, betas, eta,
                        lesion_free_scale=1
                    )

                    x0, y0 = x0s[-1], y0s[-1]

                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    self.vae = self.vae.to(device)

                    x0 = 1 / 0.18215 * x0
                    x0 = self.vae.decode([x0])[0]._sample()

                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()

                    y_h = 1 / 0.18215 * y_h
                    y_h = self.vae.decode([y_h])[0]._sample()
                    
                    x0 = x0.detach().cpu().numpy()
                    y0 = y0.detach().cpu().numpy()
                    y_h = y_h.detach().cpu().numpy()                             
                    
                    for i in tqdm(range(y0.shape[0]), desc='saving'):
                        volume_x = x0[i, 0]
                        volume_y_p = y0[i, 0]
                        volume_y_h = y_h[i, 0]

                        case_name = batch['img_name'][i] + '_' + batch['pathol_name'][i]
                        save_volume_and_slice(volume_y_h, result_dir, f"y_h_{case_name}", do_binary=False)
                        save_volume_and_slice(volume_y_p, result_dir, f"y_p_{case_name}", do_binary=False)
    '''
    '''
    def h2p_edit(self):
        """
        Healthy-to-Pathology Editing
        Directly loads healthy images + masks from txt files.
        No _pathology.nii file is required.
        """
        model_config = self.model_config
        edit_config = self.edit_config


        device = self.device
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = edit_config.eta


        result_dir = edit_config.result_dir
        os.makedirs(result_dir, exist_ok=True)


        assert edit_config.skip_type in ["uniform", "quad"], f"Unknown skip_type: {edit_config.skip_type}"


        if edit_config.skip_type == "uniform":
            skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        elif edit_config.skip_type == "quad":
            skip = max(1, int(np.sqrt(num_diffusion_timesteps) // np.sqrt(edit_config.num_sample_timesteps)))
        else:
            raise ValueError(f"Unsupported skip type: {edit_config.skip_type}")


        t_start = getattr(edit_config, "t_start", 300) # 500
        skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        denoise_t_list = range(0, t_start, skip)


        def strip_nii_ext(path: str) -> str:
            base = os.path.basename(path)
            if base.endswith(".nii.gz"):
                return base[:-7]
            if base.endswith(".nii"):
                return base[:-4]
            return os.path.splitext(base)[0]


        def load_txt_paths(txt_path: str) -> list[str]:
            with open(txt_path, "r", encoding="utf-8") as f:
                paths = [line.strip() for line in f.readlines() if line.strip()]
            if len(paths) == 0:
                raise ValueError(f"No paths found in: {txt_path}")
            return paths

        # OLD
        def load_nifti_tensor(path: str) -> torch.Tensor:
            arr = nib.load(path).get_fdata().astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr_min = float(arr.min())
            arr_max = float(arr.max())
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            return torch.from_numpy(arr).unsqueeze(0)  # [1, X, Y, Z]

        def load_image_tensor(path: str) -> torch.Tensor:
            arr = nib.load(path).get_fdata().astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


            arr_min = float(arr.min())
            arr_max = float(arr.max())


            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)


            arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(0)

        def load_mask_tensor(path: str) -> torch.Tensor:
            arr = nib.load(path).get_fdata().astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


            arr = (arr > 0).astype(np.float32)


            return torch.from_numpy(arr).unsqueeze(0)



        healthy_paths = load_txt_paths(edit_config.y_data_path)
        mask_paths = load_txt_paths(edit_config.x_data_path)


        if len(healthy_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch between healthy and mask txt files: "
                f"{len(healthy_paths)} healthy vs {len(mask_paths)} masks"
            )


        pairs = list(zip(healthy_paths, mask_paths))


        y_model = LDM_3D(
            in_channels=model_config.in_channels,
            out_channels=model_config.in_channels,
            num_channels=model_config.num_channels,
            attention_levels=model_config.attention_levels,
            num_res_blocks=model_config.num_res_blocks,
            num_head_channels=model_config.num_head_channels,
            with_conditioning=False,
            learn_sigma=model_config.learn_sigma,
        )


        y_model_path = model_config.y_model_path
        y_states = torch.load(y_model_path, map_location=device)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        y_model.eval()


        batch_size = edit_config.batch_size


        with torch.no_grad():
            with tqdm(total=len(pairs), desc="Editing", file=sys.__stdout__) as pbar:
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start:start + batch_size]

                    healthy_affines = []
                    y_h_list = []
                    x0_ori_list = []
                    img_names = []
                    pathol_names = []


                    for healthy_path, mask_path in batch_pairs:
                        if not os.path.exists(healthy_path):
                            raise FileNotFoundError(f"Healthy file not found: {healthy_path}")
                        if not os.path.exists(mask_path):
                            raise FileNotFoundError(f"Mask file not found: {mask_path}")


                        #y_h_tensor = load_nifti_tensor(healthy_path)   # [1, X, Y, Z]
                        #x0_tensor = load_nifti_tensor(mask_path)       # [1, X, Y, Z]

                        y_h_tensor = load_image_tensor(healthy_path)
                        x0_tensor = load_mask_tensor(mask_path)

                        y_h_list.append(y_h_tensor)
                        x0_ori_list.append(x0_tensor)


                        img_names.append(strip_nii_ext(healthy_path))
                        pathol_names.append(strip_nii_ext(mask_path))


                    y_h = torch.stack(y_h_list, dim=0).to(device)        # [B, 1, X, Y, Z]
                    x0_ori = torch.stack(x0_ori_list, dim=0).to(device)  # [B, 1, X, Y, Z]

                    self.vae = self.vae.to(device)

                    #y_h = self.vae.encode([y_h])[0]._sample().mul_(0.18215).to(device)
                    #x0_ori = self.vae.encode([x0_ori])[0]._sample().mul_(0.18215).to(device)

                    # healthy MRI -> latent
                    y_h = self.vae.encode([y_h])[0]._sample().mul_(0.18215).to(device)


                    # binary mask -> latent
                    x0_ori = (x0_ori > 0.5).float()
                    x0_ori = self.vae.encode([x0_ori])[0]._sample().mul_(0.18215).to(device)


                    t = torch.full((y_h.shape[0],), t_start, device=device, dtype=torch.long)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)


                    e_y = torch.randn_like(y_h)
                    yt = y_h * a.sqrt() + e_y * (1.0 - a).sqrt()


                    xt = x0_ori


                    print(
                        "[DEBUG] y_h latent:",
                        tuple(y_h.shape),
                        float(y_h.min()),
                        float(y_h.max()),
                        float(y_h.mean()),
                        float(y_h.std())
                    )


                    print(
                        "[DEBUG] x0 latent:",
                        tuple(x0_ori.shape),
                        float(x0_ori.min()),
                        float(x0_ori.max()),
                        float(x0_ori.mean()),
                        float(x0_ori.std())
                    )


                    xts, yts, x0s, y0s = denoise_h2p(
                        y_model,
                        xt,
                        yt,
                        y_h,
                        denoise_t_list,
                        betas,
                        eta,
                        lesion_free_scale=1
                    )

                    y0 = y0s[-1]


                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()


                    y_h_dec = 1 / 0.18215 * y_h
                    y_h_dec = self.vae.decode([y_h_dec])[0]._sample()


                    y0 = y0.detach().cpu().numpy().astype(np.float32)
                    y_h_dec = y_h_dec.detach().cpu().numpy().astype(np.float32)


                    for i in range(y0.shape[0]):
                        volume_y_p = y0[i, 0]
                        volume_y_h = y_h_dec[i, 0]


                        case_name = f"{img_names[i]}_{pathol_names[i]}"


                        print(
                            f"[RAW INTENSITY] {case_name} | "
                            f"y_p min={float(volume_y_p.min()):.6f}, "
                            f"max={float(volume_y_p.max()):.6f}, "
                            f"mean={float(volume_y_p.mean()):.6f}, "
                            f"std={float(volume_y_p.std()):.6f}"
                        )


                        save_volume_and_slice(volume_y_h, result_dir, f"y_h_{case_name}", do_binary=False)
                        save_volume_and_slice(volume_y_p, result_dir, f"y_p_{case_name}", do_binary=False)

                    x0, y0 = x0s[-1], y0s[-1]


                    x0 = 1 / 0.18215 * x0
                    x0 = self.vae.decode([x0])[0]._sample()


                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()


                    y_h_dec = 1 / 0.18215 * y_h
                    y_h_dec = self.vae.decode([y_h_dec])[0]._sample()


                    x0 = x0.detach().cpu().numpy().astype(np.float32)
                    y0 = y0.detach().cpu().numpy().astype(np.float32)
                    y_h_dec = y_h_dec.detach().cpu().numpy().astype(np.float32)
                    
                    # older version
                    for i in range(y0.shape[0]):
                        volume_x = x0[i, 0]
                        volume_y_p = y0[i, 0]
                        volume_y_h = y_h_dec[i, 0]


                        case_name = f"{img_names[i]}_{pathol_names[i]}"


                        save_volume_and_slice(volume_y_h, result_dir, f"y_h_{case_name}", do_binary=False)
                        save_volume_and_slice(volume_y_p, result_dir, f"y_p_{case_name}", do_binary=False)
                    
                    
                    for i in range(y0.shape[0]):
                        volume_x = x0[i, 0]
                        volume_y_p = y0[i, 0]
                        volume_y_h = y_h_dec[i, 0]


                        case_name = f"{img_names[i]}_{pathol_names[i]}"


                        print(
                            f"[RAW INTENSITY] {case_name} | "
                            f"y_p min={float(volume_y_p.min()):.6f}, "
                            f"max={float(volume_y_p.max()):.6f}, "
                            f"mean={float(volume_y_p.mean()):.6f}, "
                            f"std={float(volume_y_p.std()):.6f}"
                        )


                        save_volume_and_slice(volume_y_h, result_dir, f"y_h_{case_name}", do_binary=False)
                        save_volume_and_slice(volume_y_p, result_dir, f"y_p_{case_name}", do_binary=False)

                    pbar.update(len(batch_pairs))
                '''
    def h2p_edit(self):
        """
        Healthy-to-Pathology Editing
        Directly loads healthy images + masks from txt files.
        """
        model_config = self.model_config
        edit_config = self.edit_config


        device = self.device
        num_diffusion_timesteps = self.num_diffusion_timesteps
        betas = self.betas
        eta = edit_config.eta


        result_dir = edit_config.result_dir
        os.makedirs(result_dir, exist_ok=True)


        assert edit_config.skip_type in ["uniform", "quad"], f"Unknown skip_type: {edit_config.skip_type}"


        if edit_config.skip_type == "uniform":
            skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        elif edit_config.skip_type == "quad":
            skip = max(1, int(np.sqrt(num_diffusion_timesteps) // np.sqrt(edit_config.num_sample_timesteps)))


        t_start = getattr(edit_config, "t_start", 300)
        skip = num_diffusion_timesteps // edit_config.num_sample_timesteps
        denoise_t_list = range(0, t_start, skip)


        def strip_nii_ext(path: str) -> str:
            base = os.path.basename(path)
            if base.endswith(".nii.gz"):
                return base[:-7]
            if base.endswith(".nii"):
                return base[:-4]
            return os.path.splitext(base)[0]


        def load_txt_paths(txt_path: str) -> list[str]:
            with open(txt_path, "r", encoding="utf-8") as f:
                paths = [line.strip() for line in f.readlines() if line.strip()]
            if len(paths) == 0:
                raise ValueError(f"No paths found in: {txt_path}")
            return paths


        def load_image_tensor(path: str) -> torch.Tensor:
            arr = nib.load(path).get_fdata().astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


            arr_min = float(arr.min())
            arr_max = float(arr.max())


            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min + 1e-8)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)


            arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(0)

        '''
        def load_mask_tensor(path: str) -> torch.Tensor:
            arr = nib.load(path).get_fdata().astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = (arr > 0).astype(np.float32)
            return torch.from_numpy(arr).unsqueeze(0)
        '''

        '''
        def load_mask_tensor(path: str) -> torch.Tensor:
            seg = nib.load(path).get_fdata().astype(np.float32)
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

            # BraTS tumor core ONLY
            mask = np.logical_or(seg == 1, seg == 4).astype(np.float32)


            return torch.from_numpy(mask).unsqueeze(0)
        '''

        healthy_paths = load_txt_paths(edit_config.y_data_path)
        mask_paths = load_txt_paths(edit_config.x_data_path)
        

        def load_mask_tensor(path: str) -> torch.Tensor:
            seg = nib.load(path).get_fdata().astype(np.float32)
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)


            mask_core = np.logical_or(seg == 1, seg == 4)
            mask_grown = binary_dilation(mask_core, iterations=1)


            core_growth = 0.35


            mask = (
                (1.0 - core_growth) * mask_core.astype(np.float32)
                + core_growth * mask_grown.astype(np.float32)
            )


            return torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)



        if len(healthy_paths) != len(mask_paths):
            raise ValueError(
                f"Mismatch between healthy and mask txt files: "
                f"{len(healthy_paths)} healthy vs {len(mask_paths)} masks"
            )


        pairs = list(zip(healthy_paths, mask_paths))


        y_model = LDM_3D(
            in_channels=model_config.in_channels,
            out_channels=model_config.in_channels,
            num_channels=model_config.num_channels,
            attention_levels=model_config.attention_levels,
            num_res_blocks=model_config.num_res_blocks,
            num_head_channels=model_config.num_head_channels,
            with_conditioning=False,
            learn_sigma=model_config.learn_sigma,
        )


        y_states = torch.load(model_config.y_model_path, map_location=device)
        y_model = torch.nn.DataParallel(y_model.to(device))
        y_model.load_state_dict(y_states[0], strict=True)
        y_model.eval()


        batch_size = edit_config.batch_size


        with torch.no_grad():
            with tqdm(total=len(pairs), desc="Editing", file=sys.__stdout__) as pbar:
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start:start + batch_size]


                    y_h_list = []
                    x0_ori_list = []
                    img_names = []
                    pathol_names = []
                    healthy_affines = []


                    for healthy_path, mask_path in batch_pairs:
                        if not os.path.exists(healthy_path):
                            raise FileNotFoundError(f"Healthy file not found: {healthy_path}")
                        if not os.path.exists(mask_path):
                            raise FileNotFoundError(f"Mask file not found: {mask_path}")


                        healthy_nii = nib.load(healthy_path)
                        healthy_affines.append(healthy_nii.affine)


                        y_h_tensor = load_image_tensor(healthy_path)
                        x0_tensor = load_mask_tensor(mask_path)


                        y_h_list.append(y_h_tensor)
                        x0_ori_list.append(x0_tensor)


                        img_names.append(strip_nii_ext(healthy_path))
                        pathol_names.append(strip_nii_ext(mask_path))


                    y_h = torch.stack(y_h_list, dim=0).to(device)
                    x0_ori = torch.stack(x0_ori_list, dim=0).to(device)


                    self.vae = self.vae.to(device)


                    y_h = self.vae.encode([y_h])[0]._sample().mul_(0.18215).to(device)

                    #x0_ori = (x0_ori > 0.5).float().to(device)
                    #x0_ori = self.vae.encode([x0_ori])[0]._sample().mul_(0.18215).to(device)

                    '''
                    x0_mask_full = (x0_ori > 0.5).float().to(device)

                    x0_mask_lcg = F.interpolate(
                        x0_mask_full,
                        size=y_h.shape[2:],
                        mode="nearest",
                    )


                    x0_mask_lcg = (x0_mask_lcg > 0.5).float()


                    x0_ori = self.vae.encode([x0_mask_full])[0]._sample().mul_(0.18215).to(device)
                    '''
                    '''
                    x0_mask_full = (x0_ori > 0.5).float().to(device)

                    # Tumor-prior conditioning field:
                    # hard tumor core + weak local peritumoral field.
                    x0_soft = F.avg_pool3d(
                        x0_mask_full,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    )


                    x0_soft = torch.clamp(x0_soft, 0.0, 1.0)


                    tumor_prior = torch.clamp(
                        x0_mask_full + 0.3 * x0_soft,
                        0.0,
                        1.0,
                    )


                    # LCG still receives the hard mask, so spread is controlled.
                    x0_mask_lcg = F.interpolate(
                        x0_mask_full,
                        size=y_h.shape[2:],
                        mode="nearest",
                    )


                    x0_mask_lcg = (x0_mask_lcg > 0.5).float()


                    # The model condition receives the tumor-prior field.
                    x0_ori = self.vae.encode([tumor_prior])[0]._sample().mul_(0.18215).to(device)
                    '''

                    # Soft core-growth mask from load_mask_tensor.
                    # This preserves fractional values like 0.25 in the grown boundary.
                    x0_mask_full = torch.clamp(x0_ori, 0.0, 1.0).to(device)


                    # Hard mask only for LCG containment.
                    # This prevents the fractional grown shell from becoming the hard allowed region.
                    x0_mask_lcg_full = (x0_ori > 0.5).float().to(device)


                    # Tumor-prior conditioning field:
                    # soft grown core + weak local peritumoral field.
                    x0_soft = F.avg_pool3d(
                        x0_mask_full,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )


                    x0_soft = torch.clamp(x0_soft, 0.0, 1.0)


                    tumor_prior = torch.clamp(
                        x0_mask_full + 0.4 * x0_soft,
                        0.0,
                        1.0,
                    )


                    # LCG receives hard original core mask.
                    x0_mask_lcg = F.interpolate(
                        x0_mask_lcg_full,
                        size=y_h.shape[2:],
                        mode="nearest",
                    )


                    x0_mask_lcg = (x0_mask_lcg > 0.5).float()


                    # Model condition receives soft core-growth tumor prior.
                    x0_ori = self.vae.encode([tumor_prior])[0]._sample().mul_(0.18215).to(device)







                    t = torch.full((y_h.shape[0],), t_start, device=device, dtype=torch.long)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1, 1)


                    e_y = torch.randn_like(y_h)
                    yt = y_h * a.sqrt() + e_y * (1.0 - a).sqrt()


                    xt = x0_ori


                    print(
                        "[DEBUG] y_h latent:",
                        tuple(y_h.shape),
                        float(y_h.min()),
                        float(y_h.max()),
                        float(y_h.mean()),
                        float(y_h.std())
                    )


                    print(
                        "[DEBUG] x0 latent:",
                        tuple(x0_ori.shape),
                        float(x0_ori.min()),
                        float(x0_ori.max()),
                        float(x0_ori.mean()),
                        float(x0_ori.std())
                    )

                    xts, yts, x0s, y0s = denoise_h2p(
                        y_model,
                        xt,
                        yt,
                        y_h,
                        denoise_t_list,
                        betas,
                        eta,
                        alpha0=20.0,
                        decay=0.5,
                        lesion_free_scale=1.5,
                        x0_mask_lcg=x0_mask_lcg,
                    )

                    y0 = y0s[-1]


                    y0 = 1 / 0.18215 * y0
                    y0 = self.vae.decode([y0])[0]._sample()


                    y_h_dec = 1 / 0.18215 * y_h
                    y_h_dec = self.vae.decode([y_h_dec])[0]._sample()


                    y0 = y0.detach().cpu().numpy().astype(np.float32)
                    y_h_dec = y_h_dec.detach().cpu().numpy().astype(np.float32)


                    for i in range(y0.shape[0]):
                        volume_y_p = y0[i, 0]
                        volume_y_h = y_h_dec[i, 0]

                        '''
                        mask_full = x0_mask_full.detach().cpu().numpy()[i, 0].astype(np.float32)


                        lesion_region = mask_full > 0.5
                        brain_region = volume_y_h > 0.05
                        healthy_median = np.median(volume_y_h[brain_region])


                        tumor_floor = 0.35 * healthy_median
                        too_dark_tumor = lesion_region & (volume_y_p < tumor_floor)


                        volume_y_p[too_dark_tumor] = (
                            0.60 * volume_y_p[too_dark_tumor]
                            + 0.40 * tumor_floor
                        )'''
                        '''
                        mask_full = x0_mask_full.detach().cpu().numpy()[i, 0].astype(np.float32)

                        core = mask_full > 0.5
                        brain_region = volume_y_h > 0.05


                        healthy_median = np.median(volume_y_h[brain_region])
                        healthy_std = np.std(volume_y_h[brain_region])


                        # -----------------------------
                        # 1. Tumor core texture
                        # -----------------------------
                        noise = np.random.normal(0.0, 1.0, volume_y_p.shape).astype(np.float32)
                        texture = gaussian_filter(noise, sigma=1.0)
                        texture = (texture - texture.mean()) / (texture.std() + 1e-8)


                        tumor_base = 0.70 * healthy_median
                        tumor_texture = tumor_base + 0.10 * healthy_std * texture


                        # Keep tumor darker than normal tissue, but not black/fluid-like
                        tumor_texture = np.clip(
                            tumor_texture,
                            0.40 * healthy_median,
                            0.95 * healthy_median
                        )


                        volume_y_p[core] = (
                            0.35 * volume_y_p[core]
                            + 0.50 * tumor_texture[core]
                        )


                        # -----------------------------
                        # 2. Edema / peritumoral layer
                        # -----------------------------
                        edema = binary_dilation(core, iterations=2) & (~core) & brain_region


                        edema_soft = gaussian_filter(edema.astype(np.float32), sigma=2.0)
                        edema_soft = np.clip(edema_soft, 0.0, 1.0)


                        # Edema-like effect: subtle surrounding contrast change, not black hole
                        edema_target = 0.85 * volume_y_h


                        volume_y_p = (
                            (1.0 - 0.20 * edema_soft) * volume_y_p
                            + (0.20 * edema_soft) * edema_target
                        )


                        # -----------------------------
                        # 3. Anti-hole floor inside tumor + edema
                        # -----------------------------
                        affected = core | edema


                        intensity_floor = 0.35 * healthy_median
                        too_dark = affected & (volume_y_p < intensity_floor)


                        volume_y_p[too_dark] = (
                            0.20 * volume_y_p[too_dark]
                            + 0.80 * intensity_floor
                        )


                        # -----------------------------
                        # 4. Smooth only locally around tumor
                        # -----------------------------
                        local_zone = binary_dilation(core, iterations=2)


                        smoothed = gaussian_filter(volume_y_p, sigma=0.4)
                        volume_y_p[local_zone] = (
                            0.75 * volume_y_p[local_zone]
                            + 0.25 * smoothed[local_zone]
                        )


                        volume_y_p = np.clip(volume_y_p, 0.0, 1.0).astype(np.float32)

                        '''
                        
                        





                        case_name = f"{img_names[i]}_{pathol_names[i]}"


                        print(
                            f"[RAW INTENSITY] {case_name} | "
                            f"y_p min={float(volume_y_p.min()):.6f}, "
                            f"max={float(volume_y_p.max()):.6f}, "
                            f"mean={float(volume_y_p.mean()):.6f}, "
                            f"std={float(volume_y_p.std()):.6f}"
                        )


                        save_volume_and_slice(
                            volume_y_h,
                            result_dir,
                            f"y_h_{case_name}",
                            do_binary=False,
                            affine=healthy_affines[i]
                        )


                        save_volume_and_slice(
                            volume_y_p,
                            result_dir,
                            f"y_p_{case_name}",
                            do_binary=False,
                            affine=healthy_affines[i]
                        )


                    pbar.update(len(batch_pairs))

