# Albedo Sampler For ComfyUI

## Repository
<https://github.com/trashgraphicard/I-Will-Name-It-Later.git>

## Description
Some comfyUI nodes that assist in sampling albedo textures. The sampled albedo can be easily postprocessed into other pbr textures.

## Features
- Feature 1
	- A node that samples an image into tilable albedo texture.
- Feature 2
	- A node that leverage the selected AI model to inpaint upon the seams to make it even more flawless.

## Challenges
- Removing the lighting information from the sample image.
- Writing custom node for comfy UI
- Configure parameters for a diffusion model
- Automating mask creation for inpainting
- Support all resolution of image

## Outcomes
Ideal Outcome:
- Nodes that allows easy creation of flawless, professional level albedo texture, that can be further processed into other pbr textures

Minimal Viable Outcome:
- A node that sample an image to create albedo texture by offset the image and blending the seams (No AI)

## Milestones

- Week 1
  1. Proposal written
  2. Architect the basic workflow inside comfyUI (without my custom node)

- Week 2
  1. First node created (node that does not inviolve AI)

- Week N (Final)
  1. Second node created (This node involved AI so it might take longer.
