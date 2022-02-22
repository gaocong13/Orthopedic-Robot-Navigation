
// STD
#include <iostream>
#include <vector>

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
#include "xregImageAddPoissonNoise.h"
#include "xregProjPreProc.h"
#include "xregCIOSFusionDICOM.h"
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
#include "xregHDF5.h"

#include "xregPAODrawBones.h"
#include "xregRigidUtils.h"

#include "bigssMath.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = true;

using size_type = std::size_t;

using Pt3         = Eigen::Matrix<CoordScalar,3,1>;
using Pt2         = Eigen::Matrix<CoordScalar,2,1>;

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("This program takes femur CT and segmentation as input, using corresponding 2D and 3D femur landmarks to compute a pose. It simulates an X-ray DRR image\
               using this pose and run a 2D/3D single-view registration.");
  po.set_arg_usage("< Femur CT > < Femur Segmentation > < Femur 3D Landmark > < Femur 2D Landmark > < Example real X-ray > < output folder >");
  po.set_min_num_pos_args(5);

  po.add("fh-cen", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_STRING, "femurheadcenter",
         "Name of the femur head landmark, default is FH.")
    << "FH";

  po.add("femur-label", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "femur-label",
         "Label voxel value of the femur segmentation, default is 1.")
    << ProgOpts::uint32(1);

  po.add_backend_flags();

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

  const bool verbose = po.get("verbose");
  std::ostream& vout = po.vout();

  const std::string spec_vol_path           = po.pos_args()[0];  // Femur 3D CT nifti
  const std::string spec_seg_path           = po.pos_args()[1];  // Femur 3D Segmentation nifti
  const std::string landmark_3d_path        = po.pos_args()[2];  // 3D Landmark annotation
  const std::string landmark_2d_path        = po.pos_args()[3];  // 2D Landmark annotation
  const std::string example_real_xray       = po.pos_args()[4];  // Example real X-ray on which annotation was performed
  const std::string output_path             = po.pos_args()[5];  // Output path

  unsigned char femur_label = po.get("femur-label").as_uint32();

  const std::string femur_head_center_ld_name = po.get("femurheadcenter").as_string();

  const size_type num_views = 1; // This is single-view registration
  const size_type view_idx = 0;

  const bool use_seg = true;
  auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();
  auto so3_vars = std::make_shared<SO3OptVarsLieAlg>();

  // Start -- Load and Preprocess data
  std::cout << "[Main] Reading femur 3D anatomical landmarks from FCSV file..." << std::endl;
  auto femur_3d_fcsv = ReadFCSVFileNamePtMap(landmark_3d_path);
  ConvertRASToLPS(&femur_3d_fcsv);

  std::cout << "[Main] Reading femur 2D anatomical landmarks from FCSV file..." << std::endl;
  auto femur_2d_fcsv = ReadFCSVFileNamePtMap(landmark_2d_path);
  ConvertRASToLPS(&femur_2d_fcsv);

  xregASSERT(femur_2d_fcsv.size() > 3);

  std::vector<CIOSFusionDICOMInfo> cios_metas(1);
  ProjDataF32::Proj* example_img;

  std::tie(example_img, cios_metas[0]) = ReadCIOSFusionDICOMFloat(example_real_xray);

  UpdateLandmarkMapForCIOSFusion(cios_metas[0], &femur_2d_fcsv);

  LandMap2 pnp_2d_lands;
  pnp_2d_lands.reserve(femur_2d_fcsv.size());

  for (const auto& fcsv_kv : femur_2d_fcsv)
  {
    pnp_2d_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
  }

  vout << "[Main] Reading femur CT volume..." << std::endl;
  auto vol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spec_vol_path);

  vout << "[Main]   HU --> Att. ..." << std::endl;
  auto vol_att = HUToLinAtt(vol_hu.GetPointer());

  auto vol_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spec_seg_path);

  vout << "[Main]   cropping intensity volume tightly around labels:"
       << "\n   Femur: " << static_cast<int>(femur_label)
       << std::endl;

  vout << "[Main]   extracting femur att. volume..." << std::endl;
  auto femur_att = ApplyMaskToITKImage(vol_att.GetPointer(), vol_seg.GetPointer(), femur_label, float(0), true);

  vout << "[Main] Setting up femur head center reference frame..." << std::endl;
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> femur_singleview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto femur_fcsv_rotc = femur_3d_fcsv.find(femur_head_center_ld_name);
    Pt3 rotcenter;

    if (femur_fcsv_rotc != femur_3d_fcsv.end()){
      rotcenter = femur_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND femur head center landmark!" << std::endl;
    }

    femur_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    femur_singleview_regi_ref_frame->vol_idx = 0;
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
  }
  // End -- Load and Preprocess data

  // Start -- Set up C-arm geometry
  const auto default_cam = NaiveCamModelFromCIOSFusion(cios_metas[0], true);

  // Pose estimation by solving PnP problem using landmark annotation
  FrameTransform init_cam_to_femur = PnPPOSITAndReprojCMAES(default_cam, femur_3d_fcsv, pnp_2d_lands);

  WriteITKAffineTransform(output_path + "/femur_singleview_init_xform.h5",  init_cam_to_femur);
  // End -- Set up C-arm geometry

  // Start -- Simulate DRR projection
  ProjDataF32 proj_data;

  vout << "[Main] Simulating DRR projection data..." << std::endl;
  {
    auto ray_caster = LineIntRayCasterFromProgOpts(po);

    ray_caster->set_camera_model(default_cam);
    ray_caster->set_num_projs(1);

    ray_caster->set_volume(vol_att);
    ray_caster->allocate_resources();

    ray_caster->distribute_xform_among_cam_models(init_cam_to_femur);
    ray_caster->compute(0);

    proj_data.img = CastITKImageIfNeeded<float>(SamplePoissonProjFromAttProj(ray_caster->proj(0).GetPointer(), 5000).GetPointer());
    proj_data.cam = default_cam;
  }

  vout << "[Main] Preprocessing bone data..." << std::endl;
  ProjPreProc proj_data_preproc;
  proj_data_preproc.set_debug_output_stream(vout, verbose);
  proj_data_preproc.input_projs = { proj_data };

  proj_data_preproc();

  auto regi_proj_data = proj_data_preproc.output_projs;
  // End -- Simulate DRR projection

  // Start -- Single-view Registration
  MultiLevelMultiObjRegi regi;

  regi.set_debug_output_stream(vout, verbose);
  regi.set_save_debug_info(kSAVE_REGI_DEBUG);

  regi.vols = { femur_att };

  regi.vol_names = { "femur" };

  regi.ref_frames = { femur_singleview_regi_ref_frame };

  regi.fixed_proj_data = regi_proj_data;

  femur_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;

  regi.levels.resize(1);

  {
    regi.init_cam_to_vols = { init_cam_to_femur };

    auto& lvl = regi.levels[0];

    lvl.ds_factor = 0.25;

    lvl.fixed_imgs_to_use = { view_idx };

    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

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

    {
      auto& lvl_regi_coarse = lvl.regis[0];

      lvl_regi_coarse.mov_vols = { 0 };

      lvl_regi_coarse.static_vols = { };

      auto femur_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      femur_guess->vol_idx = 0;

      lvl_regi_coarse.init_mov_vol_poses = { femur_guess };

      lvl_regi_coarse.ref_frames = { 0 };
      {
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

        pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };

        pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20) };

        cmaes_regi->set_pop_size(150);
        cmaes_regi->set_sigma({ 3 * kDEG2RAD, 3 * kDEG2RAD, 3 * kDEG2RAD, 5, 5, 10 });

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi_coarse.regi = cmaes_regi;
      }
    }
  }

  if (kSAVE_REGI_DEBUG )
  {
    vout << "[Main] Creating registration debug info..." << std::endl;
    // Create Debug Proj H5 File
    const std::string proj_data_h5_path = output_path + "/femur_singleview_proj_data.h5";

    H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
    WriteProjDataH5(regi.fixed_proj_data, &h5);

    DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
    debug_vol_path.vol_path = spec_vol_path;

    if (use_seg)
    {
      debug_vol_path.label_vol_path = spec_seg_path;
      debug_vol_path.labels_used    = { femur_label };
    }

    regi.debug_info->vols = { debug_vol_path };

    DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
    debug_proj_path.path = proj_data_h5_path;

    regi.debug_info->fixed_projs = debug_proj_path;

    regi.debug_info->proj_pre_proc_info = proj_data_preproc.params;

    regi.debug_info->regi_names = { { "Singleview Femur" } };
  }

  vout << std::endl << "[Main] Start running single-view femur registration ..." << std::endl;
  regi.run();
  regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

  if (kSAVE_REGI_DEBUG)
  {
    vout << "[Main] Writing registration debug info to disk..." << std::endl;
    const std::string dst_debug_path = output_path + "/debug_femur.h5";
    WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
  }

  vout << "[Main] Writing registration pose to disk..." << std::endl;
  WriteITKAffineTransform(output_path + "/femur_regi_xform.h5", regi.cur_cam_to_vols[0]);
  // End -- Single-view Registration

  vout << "[Main] Reprojecting DRR to check registration result..." << std::endl;
  {
    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_model(default_cam);
    ray_caster->use_proj_store_replace_method();
    ray_caster->set_volume(regi.vols[0]);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();
    ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[0];
    ray_caster->compute(0);

    WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/femur_regi_reproj_img.png");
    WriteITKImageRemap8bpp(regi.fixed_proj_data[0].img.GetPointer(), output_path + "/regi_target_img.png");
  }

  return 0;
}
