#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.point_cloud import PointCloud


# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))


# In[3]:


sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)


# In[7]:


def run(prompt):
    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x

    # Display the sample.
    pc = sampler.output_to_point_clouds(samples)[0]

    # Produce a mesh (with vertex colors)
    ## Prereqs
    print('creating SDF model...')
    name = 'sdf'
    model = model_from_config(MODEL_CONFIGS[name], device)
    model.eval()

    print('loading SDF model...')
    model.load_state_dict(load_checkpoint(name, device))

    ## Actually make the mesh.
    mesh = marching_cubes_mesh(
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=32, # increase to 128 for resolution used in evals
        progress=True,
    )

    # Write the mesh to a PLY file to import into some other program.
    with open(f'meshes/{prompt}.ply', 'wb') as f:
        mesh.write_ply(f)
    
    return plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))


# In[11]:


import gradio as gr

txt_to_mesh = gr.Interface(fn=run, inputs="text", outputs="plot", title="Text to Mesh", description="Enter a prompt to generate a 3D mesh. The mesh will be waiting for you in meshes/").launch()


# In[ ]:




