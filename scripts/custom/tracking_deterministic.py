import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import default_sphere
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   auto_response_ssst)
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.segment.mask import median_otsu
from dipy.viz import window, actor, colormap, has_fury
import nibabel
import matplotlib.pyplot as plt

# Enables/disables interactive visualization
interactive = False

# define file paths
fimg = "../../data/input/sharepoint/ADNI/003_S_4136.nii.gz"
fbval = "../../data/input/sharepoint/ADNI/7_1027719_7_20120918154348.bval"
fbvec = "../../data/input/sharepoint/ADNI/7_1027719_7_20120918154348.bvec"

data, affine, hardi_img = load_nifti(fimg, return_img=True) 
mask, binary_mask = median_otsu(data[:, :, :, 0]) 

# read file with diffusion weigths and orientation 
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)

gtab = gradient_table(bvals, bvecs)

seed_mask = binary_mask # we want to seed from the whole brain, so use entire binary mask 
white_matter  = mask # TODO: should the white matter be the whole binary mask?
seeds = utils.seeds_from_mask(seed_mask, affine, density=1)
print(f'Init seeds: {seeds}, shape: {seeds.shape}')

response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)

csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)  
csd_fit = csd_model.fit(data, mask=white_matter)

csa_model = CsaOdfModel(gtab, sh_order=6) 
gfa = csa_model.fit(data, mask=white_matter).gfa
stopping_criterion = ThresholdStoppingCriterion(gfa, .25)

detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
    csd_fit.shm_coeff, max_angle=30., sphere=default_sphere)
streamline_generator = LocalTracking(detmax_dg, stopping_criterion, 
                                     affine=affine,
                                     seeds = seeds,
                                     max_cross=1,
                                     step_size=.5,
                                     return_all=False)
streamlines = Streamlines(streamline_generator)

##############################
# Create connectivity matrix
##############################
 
# # remove tracks shorter than 30 mm
# longer_streamlines = [t for t in streamlines if len(t)>30]

# # load atlas data
# atlas = nibabel.load('../../data/input/atlas_reg.nii.gz')
# labels = atlas.get_fdata().astype(np.uint8)

# # visualize atlas cross-section
# plt.figure()
# plt.imshow(labels[:, :, 25])
# plt.show()

# print(f'No. of unique atlas labels: {len(np.unique(labels))}')

# M = utils.connectivity_matrix(longer_streamlines, 
#                               label_volume=labels, 
#                               affine=affine, 
#                               return_mapping=True,
#                               mapping_as_streamlines=True)


# print(M)

######################################

# maybe try: dipy.align.affine_registration to find affine transformation between two 3D images


sft = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
save_trk(sft, "../../data/output/tractogram_deterministic_dg_ADNI.trk") # save tractogram 

# if has_fury:
#     scene = window.Scene()
#     scene.add(actor.line(streamlines, colormap.line_colors(streamlines)))
#     window.record(scene, out_path='../../data/output/tractogram_deterministic_dg_ADNI.png',
#                   size=(800, 800))
#     if interactive:
#         window.show(scene)