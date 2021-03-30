/*
 * Developed by Cong Gao. Email: cgao11@jhu.edu
 *
 * MIT License
 *
 * Copyright (c) 2020 Robert Grupp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <fmt/format.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregLandmarkMapUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregH5ProjDataIO.h"
#include "xregRayCastProgOpts.h"
#include "xregRayCastInterface.h"
#include "xregImgSimMetric2DPatchCommon.h"
#include "xregImgSimMetric2DGradImgParamInterface.h"
#include "xregImgSimMetric2DProgOpts.h"
#include "xregHUToLinAtt.h"
#include "xregProjPreProc.h"
#include "xregPnPUtils.h"
#include "xregMultiObjMultiLevel2D3DRegi.h"
#include "xregMultiObjMultiLevel2D3DRegiDebug.h"
#include "xregSE3OptVars.h"
#include "xregIntensity2D3DRegiExhaustive.h"
#include "xregIntensity2D3DRegiCMAES.h"
#include "xregIntensity2D3DRegiBOBYQA.h"
#include "xregRegi2D3DPenaltyFnSE3Mag.h"
#include "xregFoldNormDist.h"
#include "xregHipSegUtils.h"

using ExhaustiveRegi      = Regi2D3DExhaustive<MultiLevelMultiObjRegi::RayCaster,
MultiLevelMultiObjRegi::SimMetric>;

auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();
auto so3_vars = std::make_shared<SO3OptVarsLieAlg>();

constexpr bool kSAVE_REGI_DEBUG = true;
const bool debug_multi_view = false;

struct RegiErrorInfo
{
    float rot_error_deg;
    float trans_error;

    float rot_x_error_deg;
    float rot_y_error_deg;
    float rot_z_error_deg;

    float trans_x_error;
    float trans_y_error;
    float trans_z_error;
};

namespace{
// used to apply no regularization to the translation component
// when we are optimizing, and only changing, the rotation component
struct NullDist
{
  using Scalar = CoordScalar;

  static constexpr Scalar norm_const = 1;

  static constexpr Scalar log_norm_const = 0;

  Scalar operator()(const Scalar x) const
  {
    return 1;
  }

  Scalar log_density(const Scalar x) const
  {
    return 0;
  }
};
} // un-named

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;

  // Set up the program options

  ProgOpts po;

  jhmrPROG_OPTS_SET_COMPILE_DATE(po);
  // allow negative numbers (e.g. leading with -) positional arguments
  //po.set_allow_unrecognized_flags(true);

  po.set_help("Example driver for a multi-view femur registration using pelvis as fiducial object.");
  po.set_arg_usage("<CT 3D volume> <CT volume segmentation> <Pelvis Label> <Femur Label> "
                   "<Pelvis anatomical landmarks in 3D volume> <Femur BB landmarks in 3D volume>"
                   "<View0 anatomical landmarks & BB landmarks in projection>"
                   "<Femur entry and target points in 3D volume>"
                   "<Output Path> <CIOS Fusion 2D Image 1> ... <CIOS Fusion 2D Image N>");
  po.set_min_num_pos_args(11);

  po.add("verbose", 'v', ProgOpts::kSTORE_TRUE, "verbose", "Verbose logging to stdout")
    << false;  // default to non-verbose mode

  try
  {
    po.parse(argc, argv);
  }
  catch (const ProgOpts::Exception& e)
  {
    std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
    po.print_usage(std::cerr);
    return kEXIT_VAL_BAD_USE;
  }

  if (po.help_set())
  {
    po.print_usage(std::cout);
    po.print_help(std::cout);
    return kEXIT_VAL_SUCCESS;
  }

  boost::iostreams::stream<boost::iostreams::null_sink> null_ostream((boost::iostreams::null_sink()));
  const bool verbose = po.get("verbose");
  std::ostream& vout = verbose ? std::cout : null_ostream;

  const std::string vol_path                  = po.pos_args()[0]; // CT volume
  const std::string seg_path                  = po.pos_args()[1]; // Segmentation volume
  const LabelScalar pelvis_label              = StringCast<int>(po.pos_args()[2]); // Deafult is 1
  const LabelScalar femur_label               = StringCast<int>(po.pos_args()[3]); // Default is 2
  const std::string vol_anatld_fcsv_path      = po.pos_args()[4]; // Pelvis anatomical landmarks in 3D volume
  const std::string vol_fembb_fcsv_path       = po.pos_args()[5]; // Femur BB landmarks in 3D volume
  const std::string proj_ld_fcsv_path         = po.pos_args()[6]; // View0 anatomical landmarks & BB landmarks in projection
  const std::string vol_fempt_fcsv_path       = po.pos_args()[7]; // Femur entry and target points in 3D volume
  const std::string output_path               = po.pos_args()[8]; // Output file path
  const size_type num_views = 3;//Hard code

  std::vector<std::string> img_2d_paths(po.pos_args().begin() + 9, po.pos_args().end());

  vout << "default GPU prefs..." << std::endl;
  GPUPrefsXML gpu_prefs;

  VolPtr vol_att;
  {
    vout << "reading input CT volume..." << std::endl;
    auto vol_hu = ReadITKImageFromDisk<RayCaster::Vol>(vol_path);

    vout << "  HU --> Att. ..." << std::endl;
    auto hu2att = HounsfieldToLinearAttenuationFilter<Vol>::New();
    hu2att->SetInput(vol_hu);
    hu2att->Update();

    vol_att = hu2att->GetOutput();
  }

  vout << "reading volume segmentation..." << std::endl;
  auto vol_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(seg_path);

  vout << "extracting pelvis att. volume..." << std::endl;
  auto pelvis_vol = ApplyMaskToITKImage(vol_att.GetPointer(), vol_seg.GetPointer(), pelvis_label, PixelScalar(0), true);

  vout << "extracting femur att. volume..." << std::endl;
  auto femur_vol = ApplyMaskToITKImage(vol_att.GetPointer(), vol_seg.GetPointer(), femur_label, PixelScalar(0), true);

  vout << "reading volume landmarks from FCSV file..." << std::endl;
  auto vol_anatld_fcsv = ReadFCSVFileNamePtMap<Pt3>(vol_anatld_fcsv_path);
  vout << "  RAS --> LPS..." << std::endl;
  FromMapConvertRASToLPS(vol_anatld_fcsv.begin(), vol_anatld_fcsv.end());

  vout << "reading BB landmarks from FCSV file..." << std::endl;
  auto vol_fembb_fcsv = ReadFCSVFileNamePtMap<Pt3>(vol_fembb_fcsv_path);
  vout << "  RAS --> LPS..." << std::endl;
  FromMapConvertRASToLPS(vol_fembb_fcsv.begin(), vol_fembb_fcsv.end());

  vout << "reading femur fiducials from FCSV file..." << std::endl;
  auto vol_fempt_fcsv = ReadFCSVFileNamePtMap<Pt3>(vol_fempt_fcsv_path);
  vout << "  RAS --> LPS..." << std::endl;
  FromMapConvertRASToLPS(vol_fempt_fcsv.begin(), vol_fempt_fcsv.end());

  vout << "reading first projection landmarks FCSV file..." << std::endl;
  auto proj_ld_fcsv = ReadFCSVFileNamePtMap<Pt3>(proj_ld_fcsv_path);
  vout << "  RAS --> LPS..." << std::endl;
  FromMapConvertRASToLPS(proj_ld_fcsv.begin(), proj_ld_fcsv.end());


  ProjPreProc<PixelScalar> proj_st_pre_proc;
  proj_st_pre_proc.input_projs.resize(num_views);

  std::vector<CIOSFusionDICOMInfo> cios_metas(num_views);

  vout << "reading 2D images..." << std::endl;

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    vout << "  reading CIOS Fusion DICOM for view " << view_idx << "..." << std::endl;
    std::tie(proj_st_pre_proc.input_projs[view_idx].img, cios_metas[view_idx]) =
                                                    ReadCIOSFusionDICOM<PixelScalar>(img_2d_paths[view_idx]);

    vout << "    creating camera model..." << std::endl;
    proj_st_pre_proc.input_projs[view_idx].cam = NaiveCamModelFromCIOSFusion(
                                                    cios_metas[view_idx], true).cast<CoordScalar>();
  }

  {
    vout << "updating 2D landmarks using CIOS metadata..." << std::endl;
    UpdateLandmarkMapForCIOSFusion(cios_metas[0], proj_ld_fcsv.begin(), proj_ld_fcsv.end());

    auto& proj_lands = proj_st_pre_proc.input_projs[0].landmarks;
    proj_lands.reserve(proj_ld_fcsv.size());

    vout << "putting 2D fcsv landmarks into proj data..." << std::endl;
    for (const auto& fcsv_kv : proj_ld_fcsv)
    {
      proj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
    }
  }

  vout << "running 2D preprocessing..." << std::endl;
  proj_st_pre_proc();

  auto& projs_st_to_regi = proj_st_pre_proc.output_projs;

  vout << "running landmark based 2D/3D initialization..." << std::endl;
  const FrameTransform init_cam_0_to_vol = EstCamToWorldBruteForcePOSITCMAESRefine(projs_st_to_regi[0].cam,
                                                                                   projs_st_to_regi[0].landmarks,
                                                                                   vol_anatld_fcsv);

  const FrameTransform gt_xform = EstCamToWorldBruteForcePOSITCMAESRefine(projs_st_to_regi[0].cam,
                                                                          projs_st_to_regi[0].landmarks,
                                                                          vol_fembb_fcsv);

  vout << "gt_xform:\n" << gt_xform.matrix() << '\n';

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> pelvis_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> pelvis_multiview_regi_ref_frame;

  {
    vout << "setting up pelvis ref. frame..." << std::endl;
    // setup camera aligned reference frame, use pelvis volume center point as the origin

    const auto vol_size = pelvis_vol->GetLargestPossibleRegion().GetSize();
    itk::ContinuousIndex<double,3> center_idx;
    center_idx[0] = vol_size[0] / 2.0;
    center_idx[1] = vol_size[1] / 2.0;
    center_idx[2] = vol_size[2] / 2.0;

    itk::Point<double,3> center_pt;
    pelvis_vol->TransformContinuousIndexToPhysicalPoint(center_idx, center_pt);

    pelvis_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    pelvis_singleview_regi_ref_frame->vol_idx = 0;
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = center_pt[0];
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = center_pt[1];
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = center_pt[2];

    FrameTransform pelvis_vol_to_centered_vol = FrameTransform::Identity();
    pelvis_vol_to_centered_vol.matrix()(0,3) = -center_pt[0];
    pelvis_vol_to_centered_vol.matrix()(1,3) = -center_pt[1];
    pelvis_vol_to_centered_vol.matrix()(2,3) = -center_pt[2];

    pelvis_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(pelvis_vol_to_centered_vol, true);
  }

  // This is a ref frame for the orbital rotations - false means they'll be applied in
  // the camera world frame, which has the x-axis corresponding to orbit rotation.
  auto exhaustive_ref_frame_pel = MultiLevelMultiObjRegi::MakeStaticRefFrame(FrameTransform::Identity(), false);
  auto exhaustive_ref_frame_fem = MultiLevelMultiObjRegi::MakeStaticRefFrame(FrameTransform::Identity(), false);
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> femur_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> femur_multiview_regi_ref_frame;
  {
    Pt3 femur_pt;

    auto check_label = [&vol_seg,&femur_pt,&vol_anatld_fcsv,femur_label] (const std::string& k)
    {
      bool found = false;

      auto vol_anatld_fcsv_it = vol_anatld_fcsv.find(k);

      if (vol_anatld_fcsv_it != vol_anatld_fcsv.end())
      {
        femur_pt = vol_anatld_fcsv_it->second;

        itk::Point<CoordScalar,3> tmp_itk_pt;
        LabelVol::IndexType tmp_itk_idx;

        tmp_itk_pt[0] = femur_pt[0];
        tmp_itk_pt[1] = femur_pt[1];
        tmp_itk_pt[2] = femur_pt[2];

        vol_seg->TransformPhysicalPointToIndex(tmp_itk_pt, tmp_itk_idx);

        found = vol_seg->GetPixel(tmp_itk_idx) == femur_label;
      }

      return found;
    };

    bool found_femur_land = check_label("FH-l");
    if (found_femur_land)
    {
      vout << "  found left femoral head - will use as center of rotation for femur registration" << std::endl;
    }
    else
    {
      found_femur_land = check_label("FH-r");
      if (found_femur_land)
      {
        vout << "  found right femoral head - will use as center of rotation for femur registration" << std::endl;
      }
    }

    if (!found_femur_land)
    {
      jhmrThrow("ERROR: could not find appropriate femur landmark!!");
    }

    femur_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    femur_singleview_regi_ref_frame->vol_idx = 1;
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = femur_pt[0];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = femur_pt[1];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = femur_pt[2];

    FrameTransform femur_vol_to_fh_center = FrameTransform::Identity();
    femur_vol_to_fh_center.matrix()(0,3) = -femur_pt[0];
    femur_vol_to_fh_center.matrix()(1,3) = -femur_pt[1];
    femur_vol_to_fh_center.matrix()(2,3) = -femur_pt[2];

    femur_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(femur_vol_to_fh_center, true);
  }

  MultiLevelMultiObjRegi regi;

  regi.save_debug_info = kSAVE_REGI_DEBUG;
  regi.debug_info.vol_path       = vol_path;
  regi.debug_info.label_vol_path = seg_path;
  regi.debug_info.labels_used = { pelvis_label, femur_label };

  regi.vols = { pelvis_vol, femur_vol };
  regi.vol_names = { "Pelvis", "Femur" };

  // CG: Reference frame inputs
  regi.ref_frames = { pelvis_singleview_regi_ref_frame, exhaustive_ref_frame_pel,
                      pelvis_multiview_regi_ref_frame, femur_multiview_regi_ref_frame,
                      femur_singleview_regi_ref_frame, exhaustive_ref_frame_fem };
    // CG: readin fixed images
  regi.fixed_cam_imgs.resize(num_views);
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    regi.fixed_cam_imgs[view_idx].img = projs_st_to_regi[view_idx].img;
    regi.fixed_cam_imgs[view_idx].cam = projs_st_to_regi[view_idx].cam;
  }

  vout << "initializing number of levels and basic params..." << std::endl;

  regi.levels.resize(1);

  FrameTransformList regi_cams_to_pelvis_vol(num_views);
  FrameTransformList regi_cams_to_femur_vol(num_views);

  // Helper function for running single-view pelvis registrations
  // This will run registration for only one view
  // Level 1
  auto run_pelvis_regi = [&gpu_prefs, &vout, &regi, &regi_cams_to_pelvis_vol, &regi_cams_to_femur_vol,
                          &pelvis_singleview_regi_ref_frame, &femur_singleview_regi_ref_frame, &femur_multiview_regi_ref_frame,
                          &init_cam_0_to_vol, &output_path] (const size_type view_idx)
  {
    const bool is_first_view = view_idx == 0;

    if (is_first_view)
    {
      regi.debug_info.regi_names = { { "Pelvis0" } };
      regi.init_cam_to_vols = { init_cam_0_to_vol, FrameTransform::Identity() };
    }
    else
    {
      regi.debug_info.regi_names = { { "Pelvis-Exhau", "Pelvis-Bobyqa" } };
      regi.init_cam_to_vols = { regi_cams_to_pelvis_vol[0], regi_cams_to_femur_vol[0] };
    }

    femur_singleview_regi_ref_frame->cam_extrins = regi.fixed_cam_imgs[view_idx].cam.extrins;
    pelvis_singleview_regi_ref_frame->cam_extrins = regi.fixed_cam_imgs[view_idx].cam.extrins;

    auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();
    auto so3_vars = std::make_shared<SO3OptVarsLieAlg<double>>();

    auto& lvl = regi.levels[0];

    if (is_first_view){
      lvl.ds_factor = 0.25;
    }
    else{
      lvl.ds_factor = 0.125;
    }

    const size_type num_projs = 1;
    lvl.fixed_imgs_to_use.resize(num_projs);

    lvl.fixed_imgs_to_use = { view_idx };

    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    for (size_type proj_idx = 0; proj_idx < num_projs; ++proj_idx)
    {
      auto sm = PatchGradNCCSimMetricFromProgOpts(po);

      {
        auto* grad_sm = dynamic_cast<ImgSimMetric2DGradImgParamInterface*>(sm.get());

        grad_sm->set_smooth_img_before_sobel_kernel_radius(5);
      }

      {
        auto* patch_sm = dynamic_cast<ImgSimMetric2DPatchCommon*>(sm.get());
        xregASSERT(patch_sm);

        patch_sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
        patch_sm->set_patch_stride(1);
      }

      lvl.sim_metrics.push_back(sm);
    }

    lvl.regis.resize(is_first_view ? 1 : 2);

    if (is_first_view)
    {
      auto& lvl_regi_inten = lvl.regis[0];
      lvl_regi_inten.mov_vols = { 0 };
      lvl_regi_inten.static_vols = { };

      auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      init_guess->vol_idx = 0;
      lvl_regi_inten.init_mov_vol_poses = { init_guess };
      lvl_regi_inten.ref_frames = { 0 };
      {
        vout << "Setting up CMA-ES regi object..." << std::endl;
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        using NormPDF = NormalDist1D<CoordScalar>;

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();
        pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };
        pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(50, 50) };
        // CG: first view has larger search space

        cmaes_regi->set_pop_size(100);
        cmaes_regi->set_sigma({ 15 * kDEG2RAD, 15 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi_inten.regi = cmaes_regi;
      }
      vout << std::endl << "First view pelvis registration ..." << std::endl;
      regi.run();


      // First view femur registration

      auto& lvl = regi.levels[0];

      lvl.ds_factor = 0.125;

      const size_type num_projs = 1;
      lvl.fixed_imgs_to_use.resize(num_projs);

      lvl.fixed_imgs_to_use = { view_idx };

      lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

      for (size_type proj_idx = 0; proj_idx < num_projs; ++proj_idx)
      {
        auto sm = PatchGradNCCSimMetricFromProgOpts(po);

        {
          auto* grad_sm = dynamic_cast<ImgSimMetric2DGradImgParamInterface*>(sm.get());

          grad_sm->set_smooth_img_before_sobel_kernel_radius(5);
        }

        {
          auto* patch_sm = dynamic_cast<ImgSimMetric2DPatchCommon*>(sm.get());
          xregASSERT(patch_sm);

          patch_sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
          patch_sm->set_patch_stride(1);
        }

        lvl.sim_metrics.push_back(sm);
      }

      lvl.regis.resize(1);

      regi.debug_info.regi_names = { { "Femur0" } };
      regi.init_cam_to_vols = { regi.cur_cam_to_vols[0], regi.cur_cam_to_vols[0] };

      {
        auto& lvl_regi = lvl.regis[0];
        lvl_regi.mov_vols = { 1 };
        lvl_regi.static_vols = { 0 };

        auto pel_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        pel_init_guess->vol_idx = 0;

        auto fem_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        fem_init_guess->vol_idx = 1;

        lvl_regi.ref_frames = { 4 };

        lvl_regi.init_mov_vol_poses = { fem_init_guess };
        lvl_regi.static_vol_poses = { pel_init_guess };

        auto fem_cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
        fem_cmaes_regi->set_opt_vars(so3_vars);
        fem_cmaes_regi->set_opt_x_tol(0.01);
        fem_cmaes_regi->set_opt_obj_fn_tol(0.01);

        fem_cmaes_regi->set_pop_size(100);
        fem_cmaes_regi->set_sigma({ 17.5 * kDEG2RAD, 17.5 * kDEG2RAD, 17.5 * kDEG2RAD});

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();
        pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };
        pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(50, 50) };

        fem_cmaes_regi->set_penalty_fn(pen_fn);
        fem_cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi.regi = fem_cmaes_regi;
      }
      vout << std::endl << "First view Femur registration ..." << std::endl;
      regi.run();
      regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

      regi_cams_to_pelvis_vol[view_idx] = regi.cur_cam_to_vols[0];
      regi_cams_to_femur_vol[view_idx] = regi.cur_cam_to_vols[1];
    }
    else
    {
      auto& lvl_regi_ex = lvl.regis[0];

      lvl_regi_ex.mov_vols = { 0, 1 };
      lvl_regi_ex.static_vols = { };

      auto pel_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      pel_init_guess->vol_idx = 0;

      auto fem_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      fem_init_guess->vol_idx = 1;

      lvl_regi_ex.init_mov_vol_poses = { pel_init_guess, fem_init_guess };

      lvl_regi_ex.ref_frames = { 1, 5 };
      {
        auto ex_regi = std::make_shared<ExhaustiveRegi>();
        ex_regi->set_opt_vars(se3_vars);

        constexpr size_type kNUM_ROTS = 161;

        constexpr double kROT_INC   = 0.5 * kDEG2RAD;
        constexpr double kROT_START = -40 * kDEG2RAD;

        FrameTransformList rot_xforms(kNUM_ROTS);

        double rot_ang_rad = kROT_START;
        for (size_type rot_idx = 0; rot_idx < kNUM_ROTS; ++rot_idx, rot_ang_rad += kROT_INC)
        {
          rot_xforms[rot_idx] = EulerRotXFrame<CoordScalar>(rot_ang_rad);
        }

        ex_regi->set_cam_wrt_vols(ExhaustiveRegi::ListOfFrameTransformLists(2, rot_xforms));

        lvl_regi_ex.regi = ex_regi;
      }

      auto& lvl_regi_inten = lvl.regis[1];
      {
        lvl_regi_inten.mov_vols = { 1 };
        lvl_regi_inten.static_vols = { };

        auto fem_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        fem_init_guess->vol_idx = 1;
        lvl_regi_inten.init_mov_vol_poses = { fem_init_guess };
        lvl_regi_inten.ref_frames = { 4 };
        {
          auto bobyqa_regi = std::make_shared<Intensity2D3DRegiBOBYQA>();
          bobyqa_regi->set_opt_vars(se3_vars);
          bobyqa_regi->set_opt_x_tol(0.0001);
          bobyqa_regi->set_opt_obj_fn_tol(0.0001);
          bobyqa_regi->set_bounds({0.005 * kDEG2RAD, 0.005 * kDEG2RAD, 0.005 * kDEG2RAD, 0.005, 0.005, 0.005});

          lvl_regi_inten.regi = bobyqa_regi;
        }
      }
      regi.run();

      regi.debug_info.regi_names = { { "Pelvis-CMAES" } };
      regi.init_cam_to_vols = { regi.cur_cam_to_vols[0], regi.cur_cam_to_vols[1] };
      auto& lvl = regi.levels[0];
      {
        lvl.ds_factor = 0.25;

        lvl.fixed_imgs_to_use.resize(num_projs);
        lvl.fixed_imgs_to_use = { view_idx };

        vout << "    setting up ray caster..." << std::endl;
        lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

        for (size_type proj_idx = 0; proj_idx < num_projs; ++proj_idx)
        {
          auto sm = PatchGradNCCSimMetricFromProgOpts(po);

          {
            auto* grad_sm = dynamic_cast<ImgSimMetric2DGradImgParamInterface*>(sm.get());

            grad_sm->set_smooth_img_before_sobel_kernel_radius(5);
          }

          {
            auto* patch_sm = dynamic_cast<ImgSimMetric2DPatchCommon*>(sm.get());
            xregASSERT(patch_sm);

            patch_sm->set_patch_radius(std::lround(lvl.ds_factor * 41));
            patch_sm->set_patch_stride(1);
          }

          lvl.sim_metrics.push_back(sm);
        }

        auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();

        lvl.regis.resize(1);

        for (auto& lvl_regi : lvl.regis)
        {
          lvl_regi.mov_vols = { 0 };
          lvl_regi.static_vols = { };

          auto pel_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
          pel_init_guess->vol_idx = 0;
          lvl_regi.init_mov_vol_poses = { pel_init_guess };
        }
        auto& lvl_regi_cmaes = lvl.regis[0];

        // index into regi.ref_frames
        lvl_regi_cmaes.ref_frames = { 0 };
          // CG: setup CMAES parameters and run registration
        {
          vout << "Setting up CMA-ES regi object..." << std::endl;
          auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
          cmaes_regi->set_opt_vars(se3_vars);
          cmaes_regi->set_opt_x_tol(0.01);
          cmaes_regi->set_opt_obj_fn_tol(0.01);

          using NormPDF = NormalDist1D<CoordScalar>;

          auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();
          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(2.5 * kDEG2RAD, 2.5 * kDEG2RAD) };
          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(5, 5) };
            // CG: first view has larger search space

          cmaes_regi->set_pop_size(20);
          cmaes_regi->set_sigma({ 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 25 });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_cmaes.regi = cmaes_regi;
        }
      }
      vout << std::endl << "running CMAES pelvis registration in view " << view_idx << "..." << std::endl;
      regi.run();
      regi_cams_to_pelvis_vol[view_idx] = regi.cur_cam_to_vols[0];
      regi_cams_to_femur_vol[view_idx] = regi.cur_cam_to_vols[1];
    }
  };

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    vout << std::endl << "Start registering pelvis in view " << view_idx << "..." << std::endl;
    run_pelvis_regi(view_idx);
  }

  std::vector<CameraModel> cams;

  for (auto& pd : projs_st_to_regi)
  {
    cams.push_back(pd.cam);
  }
  cams = CreateCameraWorldUsingFiducial(cams, regi_cams_to_pelvis_vol);

  FrameTransform init_pelvis_cam_to_vols = regi_cams_to_pelvis_vol[0];
  FrameTransform pel_pre_extrins;
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    projs_st_to_regi[view_idx].cam = cams[view_idx];
    regi.fixed_cam_imgs[view_idx].cam = cams[view_idx];
  }

  // Setup Regi Level 2
  regi.levels.resize(2);

  FrameTransform init_femur_cam_to_vols = regi_cams_to_femur_vol[0] * init_pelvis_cam_to_vols.inverse();
  regi.init_cam_to_vols = { FrameTransform::Identity(), init_femur_cam_to_vols };
  regi.debug_info.regi_names = { { "Multiview Femur" }, { "Femur-Pelvis" } };
  // Femur Registration
  {
    auto& lvl = regi.levels[0];

    lvl.ds_factor = 0.25;

    lvl.fixed_imgs_to_use.resize(num_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

    vout << "    setting up ray caster..." << std::endl;
    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    {
      vout << "setting up sim metrics..." << std::endl;

      lvl.sim_metrics.resize(num_views);

      for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
      {
        auto sm = std::make_shared<PatchGradNCCImageSimilarityMetricGPU>(gpu_prefs.ctx, gpu_prefs.queue);

        sm->set_setup_vienna_cl_ctx(false);

        sm->set_smooth_img_before_sobel_kernel_radius(5);

        lvl.sim_metrics[view_idx] = sm;
      }
    }

    lvl.regis.resize(1);
    using FoldNormPDF = FoldedNormalPDF<CoordScalar>;
    using PenFn       = Regi2D3DSE3MagPenaltyFn<FoldNormPDF,NullDist>;

    {
      vout << "Setting up multiple-view femur regi..." << std::endl;

      auto& lvl_regi = lvl.regis[0];;

      lvl_regi.mov_vols    = { 1 }; // This refers to femur (moving)
      lvl_regi.static_vols = { 0 }; // This refers to pelvis and drill(static)

       auto init_guess_pelvis = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      init_guess_pelvis->vol_idx = 0;

      auto init_guess_femur = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      init_guess_femur->vol_idx = 1;

      lvl_regi.ref_frames = { 3 }; // This refers to femur (moving)

      lvl_regi.init_mov_vol_poses = { init_guess_femur };
      lvl_regi.static_vol_poses = { init_guess_pelvis };
      // Set CMAES parameters
      auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
      cmaes_regi->set_opt_vars(so3_vars);
      cmaes_regi->set_opt_x_tol(0.01);
      cmaes_regi->set_opt_obj_fn_tol(0.01);

      cmaes_regi->set_pop_size(100);
      cmaes_regi->set_sigma({ 17.5 * kDEG2RAD, 17.5 * kDEG2RAD, 17.5 * kDEG2RAD});

      auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSO3Mag>();
      pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(15 * kDEG2RAD, 15 * kDEG2RAD) };

      cmaes_regi->set_penalty_fn(pen_fn);
      cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

      lvl_regi.regi = cmaes_regi;
    }
  }

  // Pelvis & Femur Registration
  {
    auto& lvl = regi.levels[1];

    lvl.ds_factor = 0.25;

    lvl.fixed_imgs_to_use.resize(num_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

    vout << "    setting up ray caster..." << std::endl;
    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    {
      vout << "setting up sim metrics..." << std::endl;

      lvl.sim_metrics.resize(num_views);

      for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
      {
        auto sm = std::make_shared<SimMetGPU>(gpu_prefs.ctx, gpu_prefs.queue);

        sm->set_setup_vienna_cl_ctx(false);

        sm->set_smooth_img_before_sobel_kernel_radius(5);

        sm->set_patch_radius(std::lround(lvl.ds_factor * 41));

        sm->set_patch_stride(1);

        lvl.sim_metrics[view_idx] = sm;
      }
    }
    lvl.regis.resize(1);

    {
      vout << "Setting up Pelvis & Femur compound regi..." << std::endl;

      auto& lvl_regi = lvl.regis[0];
      for (auto& regi : lvl.regis)
      {
        regi.mov_vols    = { 0, 1}; // This refers to femur and pelvis (moving)
        regi.ref_frames = { 2, 3 };
        regi.static_vols = { }; // Pelvis

        auto init_guess_pelvis = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        init_guess_pelvis->vol_idx = 0;

        auto init_guess_femur = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        init_guess_femur->vol_idx = 1;

        regi.init_mov_vol_poses = { init_guess_pelvis, init_guess_femur };
      }
      auto bobyqa_regi = std::make_shared<Intensity2D3DRegiBOBYQA>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(0.0001);
      bobyqa_regi->set_opt_obj_fn_tol(0.0001);
      bobyqa_regi->set_bounds({ 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 2.5,
                                2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 2.5});

      lvl_regi.regi = bobyqa_regi;
    }
  }
  vout << "Running Femur registration & Pelvis, Femur co-registration ... " << std::endl;
  regi.run();


// ############################################   Wrap Up   ###################################################

  const FrameTransform& mv_cam_to_pelvis_vol = regi.cur_cam_to_vols[0];
  const FrameTransform& mv_cam_to_femur_vol  = regi.cur_cam_to_vols[1];
  const FrameTransform err_pelvis_xform = femur_multiview_regi_ref_frame->info.ref_frame * init_pelvis_cam_to_vols * gt_xform.inverse() * femur_multiview_regi_ref_frame->info.ref_frame.inverse();
  const FrameTransform err_femur_xform = femur_multiview_regi_ref_frame->info.ref_frame * mv_cam_to_femur_vol * init_pelvis_cam_to_vols * gt_xform.inverse() * femur_multiview_regi_ref_frame->info.ref_frame.inverse();
  const FrameTransform pre_fh_wrt_vol = femur_multiview_regi_ref_frame->info.ref_frame * mv_cam_to_femur_vol * init_pelvis_cam_to_vols;
  const FrameTransform gt_fh_wrt_vol = femur_multiview_regi_ref_frame->info.ref_frame * gt_xform;
  if (kSAVE_REGI_DEBUG)
  {
    FrameTransform cam_wrt_femur_pre  = mv_cam_to_femur_vol * init_pelvis_cam_to_vols;
    WriteITKAffineTransform(output_path+"/pre_fem_xform.h5", cam_wrt_femur_pre);
  }
  {
    auto vol_fempt_fcsv_lsc = vol_fempt_fcsv.find("LSC");
    auto vol_fempt_fcsv_lep = vol_fempt_fcsv.find("LEP");
    auto vol_fempt_fcsv_rsc = vol_fempt_fcsv.find("RSC");
    auto vol_fempt_fcsv_rep = vol_fempt_fcsv.find("REP");

    Pt3 femur_lsc_pt;
    Pt3 femur_lep_pt;
    Pt3 femur_rsc_pt;
    Pt3 femur_rep_pt;
    // Read in femur fiducial points
    if (vol_fempt_fcsv_lsc != vol_fempt_fcsv.end()){
      femur_lsc_pt = vol_fempt_fcsv_lsc->second;
    }
    else{
      vout << "ERROR: NOT FOUND LSC" << std::endl;
    }

    if (vol_fempt_fcsv_lep != vol_fempt_fcsv.end()){
      femur_lep_pt = vol_fempt_fcsv_lep->second;
    }
    else{
      vout << "ERROR: NOT FOUND LEP" << std::endl;
    }

    if (vol_fempt_fcsv_rsc != vol_fempt_fcsv.end()){
      femur_rsc_pt = vol_fempt_fcsv_rsc->second;
    }
    else{
      vout << "ERROR: NOT FOUND RSC" << std::endl;
    }

    if (vol_fempt_fcsv_rep != vol_fempt_fcsv.end()){
      femur_rep_pt = vol_fempt_fcsv_rep->second;
    }
    else{
      vout << "ERROR: NOT FOUND REP" << std::endl;
    }

    // Calculate fiducial point w.r.t cam coordinate
    Pt3 femur_lsc_cam_gt;
    Pt3 femur_lep_cam_gt;
    Pt3 femur_lsc_cam_pre;
    Pt3 femur_lep_cam_pre;
    Pt3 femur_lsc_diff;
    Pt3 femur_lep_diff;

    femur_lsc_cam_gt =  gt_xform.inverse() * femur_lsc_pt;
    femur_lep_cam_gt = gt_xform.inverse() * femur_lep_pt;
    femur_lsc_cam_pre = init_pelvis_cam_to_vols.inverse() * mv_cam_to_femur_vol.inverse() * femur_lsc_pt;
    femur_lep_cam_pre = init_pelvis_cam_to_vols.inverse() * mv_cam_to_femur_vol.inverse() * femur_lep_pt;
    femur_lsc_diff = femur_lsc_cam_gt - femur_lsc_cam_pre;
    femur_lep_diff = femur_lep_cam_gt - femur_lep_cam_pre;
    // Output fidicual point:
    vout << "femur lsc diff:" << femur_lsc_diff.norm() << std::endl;
    vout << "femur lep diff:" << femur_lep_diff.norm() << std::endl;

   // Femur Registration Error
    RegiErrorInfo e;
    RigidXformToEulerXYZAndTrans(err_femur_xform,
                                &e.rot_x_error_deg,
                                &e.rot_y_error_deg,
                                &e.rot_z_error_deg,
                                &e.trans_x_error,
                                &e.trans_y_error,
                                &e.trans_z_error);

    e.rot_x_error_deg *= kRAD2DEG;
    e.rot_y_error_deg *= kRAD2DEG;
    e.rot_z_error_deg *= kRAD2DEG;
    vout << "Femur Regi: x:" << e.trans_x_error << " y:" << e.trans_y_error << " z:" << e.trans_z_error << " rx:" << e.rot_x_error_deg << " ry:" << e.rot_y_error_deg << " rz:" << e.rot_z_error_deg << std::endl;

    std::ofstream fem_err;
    fem_err.open(output_path + "/femur_err.txt", std::ios::app);
    fem_err << "EntryPoint:" << ' ' << femur_lsc_diff.norm() << ' ' << femur_lep_diff.norm() << ' '
            << "FemurErr:"   << ' ' << e.trans_x_error       << ' ' << e.trans_y_error       << ' ' << e.trans_z_error
                             << ' ' << e.rot_x_error_deg     << ' ' << e.rot_y_error_deg     << ' ' << e.rot_z_error_deg
            << '\n';
    fem_err.close();
  }

  vout << "exiting..." << std::endl;
  return kEXIT_VAL_SUCCESS;
}
