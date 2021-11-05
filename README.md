# Registration and System Integration Software for Orthopedic Surgical Robotic System
This repository contains software programs for image-based registration, system integration and navigation tasks relating to orthopedic surgical robot applications. The repository is developed based on [xReg](https://github.com/rg2/xreg), following its general structure and routines. For more information of xReg, please visit the [wiki](https://github.com/rg2/xreg/wiki) for descriptions on the use of the library and executable programs. This repository forks the complete library support of xReg. The compilation of xReg and thirdparty libraries can be done using stand-alone clone of this repository.

This software was developed with support from Dr. [Robert Grupp](http://www.robertgrupp.com/), while conducting research under the supervision of Profs. [Mehran Armand](https://bigss.lcsr.jhu.edu), [Russell Taylor](http://www.cs.jhu.edu/~rht) and [Mathias Unberath](https://mathiasunberath.github.io/) within the [Laboratory for Computational Sensing and Robotics](https://lcsr.jhu.edu) at [Johns Hopkins University](https://www.jhu.edu).

## Library Features:
* Registration:
  * Efficient ray casters for 2D/3D registration and visualization:
    * Line integral ([CPU](lib/ray_cast/xregRayCastLineIntCPU.h) and [OpenCL](lib/ray_cast/xregRayCastLineIntOCL.h))
    * Line integral approximation via splatting ([CPU](lib/ray_cast/xregSplatLineIntCPU.h))
    * Surface rendering ([CPU](lib/ray_cast/xregRayCastSurRenderCPU.h) and [OpenCL](lib/ray_cast/xregRayCastSurRenderOCL.h))
    * Depth maps ([CPU](lib/ray_cast/xregRayCastDepthCPU.h) and [OpenCL](lib/ray_cast/xregRayCastDepthOCL.h))
    * Occluding contours ([CPU](lib/ray_cast/xregRayCastOccContourCPU.h) and [OpenCL](lib/ray_cast/xregRayCastOccContourOCL.h))
    * Sparse collision detection ([CPU](lib/ray_cast/xregRayCastSparseCollCPU.h))
    * Extendable common interface ([CPU](lib/ray_cast/xregRayCastBaseCPU.h), [OpenCL](lib/ray_cast/xregRayCastBaseOCL.h), and [more](lib/ray_cast/xregRayCastInterface.h))
  * Image similarity metrics for driving 2D/3D registrations:
    * Sum of squared differences (SSD) ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DSSDCPU.h) and [OpenCL](lib/regi/sim_metrics_2d/xregImgSimMetric2DSSDOCL.h))
    * Normalized cross correlation (NCC) ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DNCCCPU.h) and [OpenCL](lib/regi/sim_metrics_2d/xregImgSimMetric2DNCCOCL.h))
    * NCC of Sobel Gradients (Grad-NCC) ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DGradNCCCPU.h) and [OpenCL](lib/regi/sim_metrics_2d/xregImgSimMetric2DGradNCCOCL.h))
    * Gradient orientation (GO) ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DGradOrientCPU.h))
    * Gradient difference (Grad-Diff) ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DGradDiffCPU.h))
    * Patch-wise NCC (Patch-NCC) ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DPatchNCCCPU.h) and [OpenCL](lib/regi/sim_metrics_2d/xregImgSimMetric2DPatchNCCOCL.h))
    * Patch-wise Grad-NCC (Patch-Grad-NCC) ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DPatchGradNCCCPU.h) and [OpenCL](lib/regi/sim_metrics_2d/xregImgSimMetric2DPatchGradNCCOCL.h))
    * Boundary contour distance ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DBoundaryEdgesCPU.h))
    * Extendable common interface ([CPU](lib/regi/sim_metrics_2d/xregImgSimMetric2DCPU.h), [OpenCL](lib/regi/sim_metrics_2d/xregImgSimMetric2DOCL.h), and [more](lib/regi/sim_metrics_2d/xregImgSimMetric2D.h))
  * Various optimization strategies for 2D/3D intensity-based registration:
    * [CMA-ES](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiCMAES.h)
    * [Differential Evolution](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiDiffEvo.h)
    * [Exhaustive/Grid Search](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiExhaustive.h)
    * [Particle Swarm Optimization](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiPSO.h)
    * [Hill Climbing](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiHillClimb.h)
    * Wrappers around [NLOpt](https://github.com/stevengj/nlopt) routines:
      * [BOBYQA](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiBOBYQA.h)
      * [CRS](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiCRS.h)
      * [DESCH](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiDESCH.h)
      * [Variations of DIRECT](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiDIRECT.h)
      * [DISRES](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiDISRES.h)
      * [NEWUOA](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiNEWUOA.h)
      * [Nelder Mead](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiNelderMead.h)
      * [PRAXIS](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiPRAXIS.h)
      * [Sbplx](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegiSbplx.h)
    * [Extendable common interface](lib/regi/interfaces_2d_3d/xregIntensity2D3DRegi.h)
  * Regularizers for 2D/3D intensity-based registration:
    * [Rotation and translation magnitudes](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFnSE3Mag.h)
    * [Euler decomposition magnitudes](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFnSE3EulerDecomp.h)
    * [Rotation and translation magnitudes of relative pose between multiple objects](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFnRelPose.h)
    * [Relative pose difference from nominal AP pose](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFnPelvisAP.h)
    * [Landmark re-projection distances](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFnLandReproj.h)
    * [Heuristics for automatic global pelvis registration](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFnGlobalPelvis.h)
    * [Combination of regularizers](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFnCombo.h)
    * [Extendable common interface](lib/regi/penalty_fns_2d_3d/xregRegi2D3DPenaltyFn.h)
  * [Pipeline for chaining together 2D/3D registrations](lib/regi/interfaces_2d_3d/xregMultiObjMultiLevel2D3DRegi.h) (intensity-based and feature-based) for solving registration problems with multiple-resolutions and views
  * Perspective-n-Point (PnP) solvers (paired point 2D/3D)
    * [Minimization of re-projection distances](lib/regi/pnp_solvers/xregLandmark2D3DRegiReprojDistCMAES.h)
    * [POSIT](lib/regi/pnp_solvers/xregPOSIT.h)
    * [P3P using C-arm geometry assumptions](lib/regi/pnp_solvers/xregP3PCArm.h)
    * [RANSAC PnP wrapper](lib/regi/pnp_solvers/xregRANSACPnP.h)
  * [Paired Point 3D/3D](lib/regi/xregPairedPointRegi3D3D.h)
  * [3D Point Cloud to 3D Surface ICP](lib/regi/xregICP3D3D.h)
* Mesh Processing:
  * [Triangular and tetrahedral mesh representations](lib/common/xregMesh.h)
* Image/Volume Processing:
  * [Interpolation of non-uniform spaced slices](lib/image/xregVariableSpacedSlices.h)
  * Image processing operations leveraging lower-level [ITK](lib/itk) and [OpenCV](lib/opencv/xregOpenCVUtils.h) routines
  * [Conversion of Hounsfield units (HU) to linear attenuation](lib/image/xregHUToLinAtt.h)
  * [Poisson noise](lib/image/xregImageAddPoissonNoise.h)
  * [Image intensity log transform](lib/image/xregImageIntensLogTrans.h)
  * [Piecewise rigid volume warping using label maps](lib/image/xregLabelWarping.h)
* Numerical Optimization:
  * [Configurable line search implementation](lib/optim/xregLineSearchOpt.h)
  * Implementations of several derivative-free methods, including [Differential Evolution](lib/optim/xregDiffEvo.h), [Simulated Annealing](lib/optim/xregSimAnn.h), and [Particle Swarm Optimization](lib/optim/xregPSO.h)
  * [Wrapper around C implementation of CMA-ES optimization](lib/optim/xregCMAESInterface.h)
  * [Suite of test objective functions](lib/optim/xregOptimTestObjFns.h)
* Geometric Primitives and Spatial Data Structures
  * [KD-Tree for points or surfaces of arbitrary dimension](lib/spatial/xregKDTree.h)
  * [Primitives with support for intersection, etc.](lib/spatial/xregSpatialPrimitives.h)
  * Fitting primitives to data ([circle](lib/spatial/xregFitCircle.h), [plane](lib/spatial/xregFitPlane.h)) with robustness to outliers
* Spatial Transformation Utilities:
  * [Rotation](lib/transforms/xregRotUtils.h) and [rigid](lib/transforms/xregRigidUtils.h) transformation utilities, including lie group/algebra routines
  * [Perspective projection (3D to 2D)](lib/transforms/xregPerspectiveXform.h)
  * [Calculation of anatomical coordinate frames](lib/transforms/xregAnatCoordFrames.h)
  * [Point cloud manipulation](lib/transforms/xregPointCloudUtils.h)
* Visualization
  * [Interactive 3D scene plotting using VTK](lib/vtk/xregVTK3DPlotter.h)
* File I/O:
  * [Common DICOM fields](lib/file_formats/xregDICOMUtils.h)
  * [DICOM files from Siemens CIOS Fusion C-arm](lib/file_formats/xregCIOSFusionDICOM.h)
  * [Helpers for HDF5 reading/writing](lib/hdf5/xregHDF5.h)
  * Various mesh formats
  * Various image/volume formats (via ITK)
  * 3D Slicer annotations, [FCSV](lib/file_formats/xregFCSVUtils.h) and [ACSV](lib/file_formats/xregACSVUtils.h)
  * [Comma separated value (CSV) files](lib/file_formats/xregCSVUtils.h)
* Basic Math Utilities:
  * [Basic statistics](lib/basic_math/xregBasicStats.h)
  * [Distribution fitting](lib/basic_math/xregNormDistFit.h)
  * [Uniformly distributed N-D unit vector sampling](lib/basic_math/xregSampleUniformUnitVecs.h)
  * [Common interface for probability densities](lib/basic_math/xregDistInterface.h) and instances for common distributions:
    * [Normal](lib/basic_math/xregNormDist.h)
    * [Log-normal](lib/basic_math/xregLogNormDist.h)
    * [Folded normal](lib/basic_math/xregFoldNormDist.h)
* General/Common:
  * [String parsing/manipulation utilities](lib/common/xregStringUtils.h)
  * [Serialization streams](lib/common/xregStreams.h)
  * [Command line argument parsing](lib/common/xregProgOptUtils.h)
  * [Timer class for measuring runtimes with a stopwatch-like interface](lib/common/xregTimer.h)
  * [Basic filesystem utilities](lib/common/xregFilesystemUtils.h)
  * [Runtime assertions](lib/common/xregAssert.h)
* Hip Surgery:
  * [Guessing labels of bones from segmentation volumes](lib/hip_surgery/xregHipSegUtils.h)
  * [Planning and modeling of osteotomies](lib/hip_surgery/xregPAOCuts.h)
  * [Visualization of osteotomies in 3D](lib/hip_surgery/xregPAODrawBones.h)
  * [Modeling of surgical objects, such as screws and K-wires](lib/hip_surgery/xregMetalObjs.h)
  * Support for simulated data creation, including [randomized screw and K-wire shapes and poses](lib/hip_surgery/xregMetalObjSampling.h), and [volumetric data incorporating osteotomies, repositioned bones, and inserted screws and K-wires](lib/hip_surgery/xregPAOVolAfterRepo.h)

## Programs
Some of the capabilities provided by individual programs contained with the apps directory include:
* Image I/O:
  * [DICOM conversion and resampling](apps/image_io/convert_resample_dicom)
  * [Volume cropping](apps/image_io/crop_vol)
  * [Printing DICOM metadata](apps/image_io/report_dicom)
* Mesh processing:
  * [Mesh creation](apps/mesh/create_mesh)
  * [Mesh display](apps/mesh/show_mesh)
* Basic point cloud operations:
  * [Printing FCSV contents](apps/point_clouds/print_fcsv)
  * [Warping FCSV files](apps/point_clouds/xform_fcsv)
* Registration
  * [ICP for 3D point cloud to 3D surface registration](apps/mesh/sur_regi)
* General utilities for projection data:
  * [Advanced visualization of projective geometry coordinate frames with a scene of 3D objects](apps/image_io/draw_xray_scene)
  * [Remap and tile projection data for visualization](apps/image_io/remap_and_tile_proj_data)
  * [Tool for creating movie replays of 2D/3D registration processing](apps/image_io/regi2d3d_replay)
  * [Extract projection into NIFTY format (.nii/.nii.gz)](apps/image_io/extract_nii_from_proj_data)
  * [Insert landmarks (FCSV) into HDF5 projection data](apps/image_io/add_lands_to_proj_data)
* Hip Surgery: Periacetabular Osteotomy (PAO)
  * [Osteotomy planning and modeling](apps/hip_surgery/pao/create_fragment)
  * [Osteotomy 3D visualization](apps/hip_surgery/pao/draw_bones)
  * [Randomized simulation of fragment adjustments](apps/hip_surgery/pao/sample_frag_moves)
  * [Volumetric modeling of fragment adjustments](apps/hip_surgery/pao/create_repo_vol)
  * [Volumetric modeling of fragment fixation using screws and K-wires](apps/hip_surgery/pao/add_screw_kwires_to_vol)
  * [Creation of simulated fluoroscopy for 2D/3D registration experiments](apps/hip_surgery/pao/create_synthetic_fluoro)
  * Examples of 2D/3D, fluoroscopy to CT, registration
    * [Single-view pelvis registration](apps/hip_surgery/pelvis_single_view_regi_2d_3d)
    * [Multiple-view, pelvis, femur PAO fragment registration](apps/hip_surgery/pao/frag_multi_view_regi_2d_3d)

## Planned Work
Although the following capabilities currently only exist in an internal version of the xReg software, they will be incorporated into this repository at a future date:
* Executable for running a multiple-view/multiple-resolution 2D/3D registration pipeline defined using a configuration file
* Intraoperative reconstruction of PAO bone fragments
* Utilities for creation and manipulation of statistical shape models
* Shape completion from partial shapes and statistical models
* More point cloud manipulation utilities
* Python bindings, conda integration
* And more...

## Dependencies
* C++ 11 compatible compiler
* External libraries (compatible versions are listed):
  * OpenCL (1.x) (typically provided with your graphics drivers or CUDA SDK)
  * [Intel Threading Building Blocks (TBB)](https://github.com/oneapi-src/oneTBB) (20170919oss)
  * [Boost](https://www.boost.org) (header only) (1.65)
  * [Eigen3](http://eigen.tuxfamily.org) (3.3.4)
  * [fmt](https://fmt.dev) (5.3.0)
  * [NLOpt](https://github.com/stevengj/nlopt) (2.5.0)
  * [ITK](https://itk.org) (4.13.2)
  * [VTK](https://vtk.org) (7.1.1)
  * [OpenCV](https://opencv.org) (3.2.0)
  * [ViennaCL](http://viennacl.sourceforge.net) (1.7.1)
  * Optional: [ffmpeg](https://ffmpeg.org) is used for writing videos when it is found in the system path. The OpenCV video writer is used if ffmpeg is not available.

## Building
A standard CMake configure/generate process is used.
It is recommended to generate Ninja build files for fast and efficient compilation.
An example script for building all dependencies (except OpenCL) and the xReg repository is also provided [here](example_build_script).
The [docker](docker) directory demonstrates how Docker may be used to build the software.

## Acknowledgement
Development of this software results in the following publication references:
```
C. Gao et al., "Fiducial-Free 2D/3D Registration for Robot-Assisted Femoroplasty," in IEEE Transactions on Medical Robotics and Bionics, vol. 2, no. 3, pp. 437-446, Aug. 2020, doi: 10.1109/TMRB.2020.3012460.
----------------------------------------------------------------------
@ARTICLE{9151197,  author={C. {Gao} and A. {Farvardin} and R. B. {Grupp} and M. {Bakhtiarinejad} and L. {Ma} and M. {Thies} and M. {Unberath} and R. H. {Taylor} and M. {Armand}},  journal={IEEE Transactions on Medical Robotics and Bionics},   title={Fiducial-Free 2D/3D Registration for Robot-Assisted Femoroplasty},   year={2020},  volume={2},  number={3},  pages={437-446},  doi={10.1109/TMRB.2020.3012460}}
```
```
Cong Gao, Robert B. Grupp, Mathias Unberath, Russell H. Taylor, Mehran Armand, "Fiducial-free 2D/3D registration of the proximal femur for robot-assisted femoroplasty," Proc. SPIE 11315, Medical Imaging 2020: Image-Guided Procedures, Robotic Interventions, and Modeling, 113151C (16 March 2020); https://doi.org/10.1117/12.2550992
----------------------------------------------------------------------
@inproceedings{10.1117/12.2550992,
author = {Cong Gao and Robert B. Grupp and Mathias Unberath and Russell H. Taylor and Mehran Armand},
title = {{Fiducial-free 2D/3D registration of the proximal femur for robot-assisted femoroplasty}},
volume = {11315},
booktitle = {Medical Imaging 2020: Image-Guided Procedures, Robotic Interventions, and Modeling},
editor = {Baowei Fei and Cristian A. Linte},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {350 -- 355},
keywords = {2D/3D Registration, Femur Registration, X-ray Navigation, Femoroplasty},
year = {2020},
doi = {10.1117/12.2550992},
URL = {https://doi.org/10.1117/12.2550992}
}
```
