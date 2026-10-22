# Controllable Text-to-Image Generation with Stable Diffusion and CLIP-Based Evaluation

How much do the main Stable Diffusion settings (number of steps, guidance scale, image size, negative prompts) actually change **how well an image follows its prompt**?

This project generates images with a pretrained Stable Diffusion model through Hugging Face `diffusers` and measures prompt adherence with **CLIP score**, changing one setting at a time with a fixed seed and a fixed prompt set, so that settings are compared fairly.

## Key results

Mean CLIP score over 4 evaluation prompts (higher = closer match between image and prompt).

| Experiment | Setting | Mean CLIP score |
|---|---|:-:|
| Inference steps | 10 | 27.93 |
| | 25 | 28.63 |
| | 50 | 28.81 |
| | 100 | **28.84** |
| Guidance scale | 3.0 | 26.86 |
| | **7.5** | **28.51** |
| | 12.0 | 28.42 |
| Resolution | 512 × 512 | **28.62** |
| | 512 × 768 (portrait) | 25.66 |
| | 768 × 512 (landscape) | 28.31 |
| Negative prompt | off | **28.51** |
| | on | 27.26 |

**Seed variation (4 seeds per prompt):** the standard deviation per prompt ranged from 0.45 to 2.18 CLIP points. This is the noise floor: a difference between two settings smaller than this is not meaningful.

## Findings

1. **More steps help only up to about 25.** The score rises from 27.93 (10 steps) to 28.63 (25 steps), then barely moves (28.84 at 100 steps). Beyond 25 steps you pay 4× the compute for almost nothing.
2. **Guidance 7.5 is the sweet spot.** Low guidance (3.0) lets the model drift from the prompt (26.86); very high guidance (12.0) does not improve adherence further.
3. **Portrait resolution hurts.** 512 × 768 drops the score by about 3 points (25.66), while landscape 768 × 512 stays close to square. The model was trained mostly at 512 × 512, and tall images change the composition it produces.
4. **A generic negative prompt did not improve adherence** (27.26 vs 28.51). Steering away from "ugly, distorted, blurry" can make images look cleaner without making them match the prompt better.
5. Several of these gaps are close to the seed noise, so they are **trends, not proof**. A larger prompt set is needed for firm conclusions.

## Setup

| Item | Choice |
|---|---|
| Generator | `dreamlike-art/dreamlike-diffusion-1.0` (a Stable Diffusion 1.x fine-tune) via `StableDiffusionPipeline` |
| Scheduler | DDIM |
| Evaluator | CLIP `openai/clip-vit-large-patch14` |
| CLIP score | 100 × cosine similarity between image and text embeddings, clipped at 0 |
| Seed | 42 for every run (image *i* of a prompt uses seed 42 + *i*) |
| Precision | float16 on GPU |

**Evaluation prompts**
- a teacher walking inside a classroom
- a red vintage car parked on a cobblestone street at sunset
- a bowl of fresh fruit on a wooden kitchen table
- a lighthouse on a rocky coast during a storm

Steps were fixed at 30 for the guidance, resolution and negative-prompt experiments. Negative prompt used: *"ugly, distorted, blurry, low quality, deformed"*.

## How to run

1. Open the notebook in Google Colab and choose a **T4 GPU** runtime.
2. Run all cells (about 10 to 15 minutes). The notebook installs `diffusers`, `transformers` and `accelerate`.
3. All scores are saved to `clip_scores.csv`, and a bar chart compares the settings.

To test your own prompt, call `generate("your prompt", num_inference_steps=25, guidance_scale=7.5)`.

## What CLIP score does and does not measure

CLIP score measures **text-image alignment only**. A blurry image that clearly shows the right objects can still score well, so the generated images should be checked alongside the numbers. A fuller study would add an image-quality metric such as FID against a reference set, and many more prompts.

## Main libraries

PyTorch · Hugging Face Diffusers · Transformers · Accelerate · pandas · Matplotlib

## Author

**Sadaf Zehra**, MS Data Science, NED University of Engineering and Technology
[GitHub](https://github.com/SadafZehra) · [LinkedIn](https://www.linkedin.com/in/sadaf-zehra-796858230) · [ORCID](https://orcid.org/0009-0009-7530-9006)
