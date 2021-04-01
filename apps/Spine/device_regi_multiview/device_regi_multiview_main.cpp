
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

FrameTransform ConvertSlicerToITK(std::vector<float> slicer_vec){
  FrameTransform RAS2LPS;
  RAS2LPS(0, 0) = -1; RAS2LPS(0, 1) = 0; RAS2LPS(0, 2) = 0; RAS2LPS(0, 3) = 0;
  RAS2LPS(1, 0) = 0;  RAS2LPS(1, 1) = -1;RAS2LPS(1, 2) = 0; RAS2LPS(1, 3) = 0;
  RAS2LPS(2, 0) = 0;  RAS2LPS(2, 1) = 0; RAS2LPS(2, 2) = 1; RAS2LPS(2, 3) = 0;
  RAS2LPS(3, 0) = 0;  RAS2LPS(3, 1) = 0; RAS2LPS(3, 2) = 0; RAS2LPS(3, 3) = 1;

  FrameTransform ITK_xform;
  for(size_type idx=0; idx<4; ++idx)
  {
    for(size_type idy=0; idy<3; ++idy)
    {
      ITK_xform(idy, idx) = slicer_vec[idx*3+idy];
    }
    ITK_xform(3, idx) = 0.0;
  }
  ITK_xform(3, 3) = 1.0;

  ITK_xform= RAS2LPS * ITK_xform * RAS2LPS;

  float tmp1, tmp2, tmp3;
  tmp1 = ITK_xform(0,0)*ITK_xform(0,3) + ITK_xform(0,1)*ITK_xform(1,3) + ITK_xform(0,2)*ITK_xform(2,3);
  tmp2 = ITK_xform(1,0)*ITK_xform(0,3) + ITK_xform(1,1)*ITK_xform(1,3) + ITK_xform(1,2)*ITK_xform(2,3);
  tmp3 = ITK_xform(2,0)*ITK_xform(0,3) + ITK_xform(2,1)*ITK_xform(1,3) + ITK_xform(2,2)*ITK_xform(2,3);

  ITK_xform(0,3) = -tmp1;
  ITK_xform(1,3) = -tmp2;
  ITK_xform(2,3) = -tmp3;

  return ITK_xform;
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Compute 3D Polaris Position of Snake Tip Jig");
  po.set_arg_usage("<Source Bayview H5 root path> <Bayview H5 Debug file root path> <Bayview slicer root path> <Experiment list file path> <Result file path> <Top landmark path> <result path>");
  po.set_min_num_pos_args(4);

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

  const std::string deviceld_2d_fcsv_path      = po.pos_args()[0];  // 2D device landmarks path
  const std::string femurld_2d_fcsv_path       = po.pos_args()[1];  // 2D femur landmarks path
  const std::string init_xform_folder          = po.pos_args()[2];
  const std::string output_path                = po.pos_args()[3];
  const std::string img_path                   = po.pos_args()[4];

  std::vector<std::string> img_ID_list(po.pos_args().begin()+5, po.pos_args().end()); // First 3 are device; Last 3 are femur

  const size_type num_views = 3;//Hard code

  const std::string device_3d_fcsv_path = meta_data_path + "/Device3Dlandmark.fcsv";
  const std::string device_3d_bb_fcsv_path = meta_data_path + "/Device3Dbb.fcsv";
  const std::string devicevol_path = meta_data_path + "/Device_crop_CT.nii.gz";
  const std::string deviceseg_path = meta_data_path + "/Device_crop_seg.nii.gz";

  std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);


  auto device_2d_fcsv = ReadFCSVFileNamePtMap(deviceld_2d_fcsv_path);
  ConvertRASToLPS(&device_2d_fcsv);

  const bool use_seg = true;
  using UseCurEstForInit = MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst;


  const bool use_seg = true;
  auto device_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(deviceseg_path);

  vout << "reading device volume..." << std::endl; // We only use the needle metal part
  auto devicevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(devicevol_path);

  vout << "  HU --> Att. ..." << std::endl;

  auto devicevol_att = HUToLinAtt(devicevol_hu.GetPointer());

  unsigned char device_label = 1;

  auto device_vol = ApplyMaskToITKImage(devicevol_att.GetPointer(), device_seg.GetPointer(), device_label, float(0), true);

  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> device_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> device_multiview_regi_ref_frame;
  {
    vout << "setting up device ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal device volume center point as the origin

    itk::ContinuousIndex<double,3> center_idx;

    auto device_fcsv_rotc = device_3d_fcsv.find("RotCenter");
    Pt3 device_rotcenter;

    if (device_fcsv_rotc != device_3d_fcsv.end()){
      device_rotcenter = device_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND device ROT CENTER" << std::endl;
    }

    device_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    device_singleview_regi_ref_frame->vol_idx = 0;
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = device_rotcenter[0];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = device_rotcenter[1];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = device_rotcenter[2];

    FrameTransform device_vol_to_centered_vol = FrameTransform::Identity();
    device_vol_to_centered_vol.matrix()(0,3) = -device_rotcenter[0];
    device_vol_to_centered_vol.matrix()(1,3) = -device_rotcenter[1];
    device_vol_to_centered_vol.matrix()(2,3) = -device_rotcenter[2];

    device_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(device_vol_to_centered_vol, true);
  }

  ProjPreProc proj_pre_proc;
  proj_pre_proc.input_projs.resize(num_views);
  std::vector<CIOSFusionDICOMInfo> device_cios_metas(num_views);
  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    std::string img_path_ID = img_path + "/" + img_ID_list[view_idx];
    {
      std::tie(proj_pre_proc.input_projs[view_idx].img, device_cios_metas[view_idx]) = ReadCIOSFusionDICOMFloat(img_path_ID);
      proj_pre_proc.input_projs[view_idx].cam = NaiveCamModelFromCIOSFusion(device_cios_metas[view_idx], true);
    }
  }

  {
    UpdateLandmarkMapForCIOSFusion(device_cios_metas[0], &device_2d_fcsv);

    auto& device_proj_lands = proj_pre_proc.input_projs[0].landmarks;
    device_proj_lands.reserve(device_2d_fcsv.size());

    for (const auto& fcsv_kv : device_2d_fcsv)
    {
      device_proj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
    }
  }

  proj_pre_proc();

  auto& projs_to_regi = proj_pre_proc.output_projs;

  FrameTransformList init_device_xform;

  init_device_xform.reserve(num_views);

  for(size_type view_idx = 0;view_idx < num_views; ++view_idx)
  {
    const std::string device_xform_file_path = init_xform_folder + "/device_regi_xform" + img_ID_list[view_idx] + ".h5";
    init_device_xform[view_idx] = ReadITKAffineTransformFromFile(device_xform_file_path);
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  std::vector<CameraModel> cams_devicefid;
  for (auto& pd : projs_to_regi)
  {
    cams_devicefid.push_back(pd.cam);
  }

  // Using device regi as fiducial
  CreateCameraWorldUsingFiducial(cams_devicefid, regi_cams_to_device_vol);

  FrameTransform init_device_cam_to_vols = regi_cams_to_device_vol[0];

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    projs_to_regi[view_idx].cam = cams_devicefid[view_idx];
    regi.fixed_proj_data[view_idx].cam = cams_devicefid[view_idx];
  }

  MultiLevelMultiObjRegi regi;

  regi.set_debug_output_stream(vout, verbose);
  regi.set_save_debug_info(kSAVE_REGI_DEBUG);
  regi.vols = { device_vol };
  regi.vol_names = { "Device"};

  regi.ref_frames = { device_multiview_regi_ref_frame };

  const size_type view_idx = 0;

  regi.fixed_proj_data = proj_pre_proc.output_projs;

  regi.levels.resize(1);

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();

  // Multi-view Device Registration
  {
    auto& lvl = regi.levels[0];

    lvl.ds_factor = 0.25;

    lvl.fixed_imgs_to_use.resize(num_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

    vout << "    setting up ray caster..." << std::endl;
    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    vout << "    setting up sim metrics..." << std::endl;
    lvl.sim_metrics.reserve(num_views);

    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
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
      vout << "Setting up multiple-view device regi..." << std::endl;

      auto& lvl_regi = lvl.regis[0];;

      lvl_regi.mov_vols    = { 0 }; // This refers to femur (moving)
      lvl_regi.static_vols = {  }; // This refers to pelvis and device(static)

      auto device_init_guess = std::make_shared<UseCurEstForInit>();
      device_init_guess->vol_idx = 0;

      lvl_regi.ref_frames = { 0 };

      lvl_regi.init_mov_vol_poses = { device_init_guess };

      // Set CMAES parameters
      auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(0.01);
      cmaes_regi->set_opt_obj_fn_tol(0.01);

      auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

      pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(5 * kDEG2RAD, 5 * kDEG2RAD) };
      pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(10, 10) };

      cmaes_regi->set_pop_size(50);
      cmaes_regi->set_sigma({ 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 5 });

      cmaes_regi->set_penalty_fn(pen_fn);
      cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

      lvl_regi.regi = cmaes_regi;
    }
  }

  if (kSAVE_REGI_DEBUG )
  {
    vout << "  setting regi debug info..." << std::endl;

    DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
    debug_vol_path.vol_path = devicevol_path;

    if (use_seg)
    {
      debug_vol_path.label_vol_path = deviceseg_path;
      debug_vol_path.labels_used    = { device_label };
    }

    regi.debug_info->vols = { debug_vol_path };

    DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
    debug_proj_path.path = proj_data_h5_path;
    // debug_proj_path.projs_used = { view_idx };

    regi.debug_info->fixed_projs = debug_proj_path;

    regi.debug_info->proj_pre_proc_info = proj_pre_proc.params;

    regi.debug_info->regi_names = { { "Multiview DeviceRegi" } };
  }

  vout << std::endl << " ************  Running Multi-view Device Registration ... " << std::endl;
  regi.run();

  if (kSAVE_REGI_DEBUG)
  {
    vout << "writing debug info to disk..." << std::endl;
    const std::string dst_debug_path = output_path + "/debug_device_mv.h5";  
    WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
  }

  FrameTransform device_regi_xform = regi.cur_cam_to_vols[0];
  
  vout << "saving transformations ..." << std::endl;

  WriteITKAffineTransform(output_path + "/device_regi_xform.h5", device_regi_xform);

  return 0;
}
