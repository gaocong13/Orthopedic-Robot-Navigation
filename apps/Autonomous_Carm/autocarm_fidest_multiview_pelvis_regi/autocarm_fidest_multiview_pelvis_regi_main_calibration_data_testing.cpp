
// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregITKMathOps.h"
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

  po.set_help("Multi-view pelvis registration using C-arm geometry estimated by Carm fiducial.");
  po.set_arg_usage("< Meta data path > < pelvis X-ray name txt file > < calibration X-ray name txt file > < 2D landmark pelvis landmark annotation folder > "
                   "< pelvis X-ray image DCM folder > < calibration X-ray image DCM folder > < handeyeX > < Tracker data folder > < output folder >");
  po.set_min_num_pos_args(5);

  po.add("pelvis-label", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "pelvis-label",
         "Label voxel value of the pelvis segmentation, default is 1.")
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

  const std::string meta_data_path               = po.pos_args()[0];  // Meta Data Folder containing CT, Segmentation and 3D landmark annotations
  const std::string pelvis_xray_id_txt_path      = po.pos_args()[1];  // Experiment list file path, containing name of the pelvis X-ray image
  const std::string calibration_xray_id_txt_path = po.pos_args()[2];  // Experiment list file path, containing name of the calibration X-ray image
  const std::string landmark2d_root_path         = po.pos_args()[3];  // 2D Landmark folder path, containing annotations with the name of "pelvis***.fcsv"
  const std::string pelvis_dicom_path            = po.pos_args()[4];  // Pelvis Dicom X-ray image folder path
  const std::string calibration_dicom_path       = po.pos_args()[5];  // Dicom X-ray image folder path
  const std::string handeyeX_path                = po.pos_args()[6];  // Path to handeyeX matrix
  const std::string pelvis_regi_xform_path       = po.pos_args()[7];  // Path to pelvis registration matrix
  const std::string calibraion_tracker_path      = po.pos_args()[8];  // Path to calibration tracker data
  const std::string output_path                  = po.pos_args()[9];  // Output path

  const std::string spec_vol_path = meta_data_path + "/Spec22-2181-CT-Bone-1mm.nii.gz";
  const std::string spec_seg_path = meta_data_path + "/Spec22-2181-Seg-Bone-1mm.nii.gz";
  const std::string pelvis_3d_fcsv_path = meta_data_path + "/pelvis_3D_landmarks.fcsv";

  unsigned char pelvis_label = po.get("pelvis-label").as_uint32();

  const size_type num_pelvis_views = 1; // This is single-view pelvis registration

  const bool use_seg = true;
  auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();
  auto so3_vars = std::make_shared<SO3OptVarsLieAlg>();

  std::cout << "reading pelvis anatomical landmarks from FCSV file..." << std::endl;
  auto pelvis_3d_fcsv = ReadFCSVFileNamePtMap(pelvis_3d_fcsv_path);
  ConvertRASToLPS(&pelvis_3d_fcsv);

  vout << "reading pelvis CT volume..." << std::endl;
  auto vol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spec_vol_path);

  vout << "  HU --> Att. ..." << std::endl;
  auto vol_att = HUToLinAtt(vol_hu.GetPointer());

  auto vol_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spec_seg_path);

  vout << "cropping intensity volume tightly around labels:"
       << "\n  Pelvis: " << static_cast<int>(pelvis_label)
       << std::endl;

  vout << "extracting pelvis att. volume..." << std::endl;
  auto pelvis_vol = ApplyMaskToITKImage(vol_att.GetPointer(), vol_seg.GetPointer(), pelvis_label, float(0), true);

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> pelvis_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> pelvis_multiview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto pelvis_fcsv_rotc = pelvis_3d_fcsv.find("pelvis-cen");
    Pt3 rotcenter;

    if (pelvis_fcsv_rotc != pelvis_3d_fcsv.end()){
      rotcenter = pelvis_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND pelvis rotation center" << std::endl;
    }

    pelvis_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    pelvis_singleview_regi_ref_frame->vol_idx = 0;
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];

    FrameTransform pelvis_vol_to_centered_vol = FrameTransform::Identity();
    pelvis_vol_to_centered_vol.matrix()(0,3) = -rotcenter[0];
    pelvis_vol_to_centered_vol.matrix()(1,3) = -rotcenter[1];
    pelvis_vol_to_centered_vol.matrix()(2,3) = -rotcenter[2];

    pelvis_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(pelvis_vol_to_centered_vol, true);
  }

  // Read Pelvis X-ray Image
  std::vector<std::string> pelvis_xray_ID_list;
  int pelvis_lineNumber = 0;
  /* Read exp ID list from file */
  {
    std::ifstream expIDFile(pelvis_xray_id_txt_path);
    // Make sure the file is open
    if(!expIDFile.is_open()) throw std::runtime_error("Could not open pelvis X-ray ID file");

    std::string line, csvItem;

    while(std::getline(expIDFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          pelvis_xray_ID_list.push_back(csvItem);
      }
      pelvis_lineNumber++;
    }
  }

  if(pelvis_lineNumber!=pelvis_xray_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  if(pelvis_lineNumber!=num_pelvis_views) throw std::runtime_error("More than One image parsed!!!");

  // Read Calibration X-ray Image
  std::vector<std::string> calibration_xray_ID_list;
  int calibration_lineNumber = 0;
  /* Read exp ID list from file */
  {
    std::ifstream expIDFile(calibration_xray_id_txt_path);
    // Make sure the file is open
    if(!expIDFile.is_open()) throw std::runtime_error("Could not open calibration X-ray ID file");

    std::string line, csvItem;

    while(std::getline(expIDFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          calibration_xray_ID_list.push_back(csvItem);
      }
      calibration_lineNumber++;
    }
  }

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  const std::string pel_xray_ID                   = pelvis_xray_ID_list[0];
  const std::string pelvis_landmark_2d_fcsv_path  = landmark2d_root_path + "/pelvis" + pel_xray_ID + ".fcsv";
  const std::string pelvis_img_path               = pelvis_dicom_path + "/" + pel_xray_ID;

  const std::string cal_xray_01_ID       = calibration_xray_ID_list[0];
  const std::string cal_xray_02_ID       = calibration_xray_ID_list[1];
  const std::string cal_img_01_path      = calibration_dicom_path + "/" + cal_xray_01_ID;
  const std::string cal_img_02_path      = calibration_dicom_path + "/" + cal_xray_02_ID;

  // Find relative geometry between cal img 01 and 02
  auto handeyeX_xform = ReadITKAffineTransformFromFile(handeyeX_path);
  auto pelvis_regi_xform = ReadITKAffineTransformFromFile(pelvis_regi_xform_path);
  FrameTransformList cal_RB4_wrt_CarmFid_xform_list;
  FrameTransformList cal_pnp_xform_list;
  FrameTransformList rel_carm_xform_list;

  size_type num_cal_views = calibration_xray_ID_list.size();

  vout << "Reading calibration tracker data..." << std::endl;
  for(size_type cal_id = 0; cal_id < num_cal_views; ++cal_id)
  {
    const std::string RB4_xform_path        = calibraion_tracker_path + "/" + calibration_xray_ID_list[cal_id] + "/RB4.h5";
    H5::H5File h5_RB4(RB4_xform_path, H5F_ACC_RDWR);

    H5::Group RB4_transform_group           = h5_RB4.openGroup("TransformGroup");
    H5::Group RB4_group0                    = RB4_transform_group.openGroup("0");
    std::vector<float> RB4_slicer           = ReadVectorH5Float("TranformParameters", RB4_group0);

    FrameTransform RB4_xform = ConvertSlicerToITK(RB4_slicer);

    const std::string CarmFid_xform_path    = calibraion_tracker_path + "/" + calibration_xray_ID_list[cal_id] + "/BayviewSiemensCArm.h5";
    H5::H5File h5_CarmFid(CarmFid_xform_path, H5F_ACC_RDWR);

    H5::Group CarmFid_transform_group       = h5_CarmFid.openGroup("TransformGroup");
    H5::Group CarmFid_group0                = CarmFid_transform_group.openGroup("0");
    std::vector<float> CarmFid_slicer       = ReadVectorH5Float("TranformParameters", CarmFid_group0);

    FrameTransform CarmFid_xform = ConvertSlicerToITK(CarmFid_slicer);

    FrameTransform RB4_wrt_CarmFid = RB4_xform.inverse() * CarmFid_xform;
    cal_RB4_wrt_CarmFid_xform_list.push_back(RB4_wrt_CarmFid);

    FrameTransform rel_carm_xform = pelvis_regi_xform * handeyeX_xform.inverse() * cal_RB4_wrt_CarmFid_xform_list[0].inverse() * cal_RB4_wrt_CarmFid_xform_list[cal_id] * handeyeX_xform;
    rel_carm_xform_list.push_back(rel_carm_xform);
  }

  std::cout << "Running..." << pel_xray_ID << std::endl;
  auto pelvis_landmark_2d_fcsv = ReadFCSVFileNamePtMap(pelvis_landmark_2d_fcsv_path);
  ConvertRASToLPS(&pelvis_landmark_2d_fcsv);

  xregASSERT(pelvis_landmark_2d_fcsv.size() > 3);

  ProjPreProc proj_pre_proc;
  proj_pre_proc.input_projs.resize(num_cal_views);

  ProjDataF32 pel_img;
  std::vector<CIOSFusionDICOMInfo> pel_cios_metas(1), cal_cios_metas(1);
  {
    std::tie(proj_pre_proc.input_projs[0].img, pel_cios_metas[0]) = ReadCIOSFusionDICOMFloat(pelvis_img_path);
    proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(pel_cios_metas[0], true);
  }

  for(size_type view_idx = 1; view_idx < num_cal_views; ++view_idx)
  {
    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_model(default_cam);
    ray_caster->use_proj_store_replace_method();
    ray_caster->set_volume(vol_att);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();
    ray_caster->xform_cam_to_itk_phys(0) = rel_carm_xform_list[view_idx];
    ray_caster->compute(0);

    proj_pre_proc.input_projs[view_idx].img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 5000);
    proj_pre_proc.input_projs[view_idx].cam = NaiveCamModelFromCIOSFusion(pel_cios_metas[0], true);
  }

  proj_pre_proc();
  auto& projs_to_regi = proj_pre_proc.output_projs;

  std::vector<CameraModel> orig_cams;
  for (auto& pd : projs_to_regi)
  {
    orig_cams.push_back(pd.cam);
  }

  // Using device regi as fiducial
  auto cams_devicefid = CreateCameraWorldUsingFiducial(orig_cams, rel_carm_xform_list);

  const size_type view_idx = 0;

  FrameTransform init_pelvis_xform = FrameTransform::Identity();

  MultiLevelMultiObjRegi regi;

  regi.set_debug_output_stream(vout, verbose);
  regi.set_save_debug_info(kSAVE_REGI_DEBUG);
  regi.vols = { pelvis_vol };
  regi.vol_names = { "pelvis" };

  regi.ref_frames = { pelvis_multiview_regi_ref_frame };

  regi.fixed_proj_data = proj_pre_proc.output_projs;

  for (size_type view_idx = 0; view_idx < num_cal_views; ++view_idx)
  {
    regi.fixed_proj_data[view_idx].cam = cams_devicefid[view_idx];
  }

  regi.levels.resize(1);

  // Pelvis Registration First
  {
    regi.init_cam_to_vols = { init_pelvis_xform };

    auto& lvl = regi.levels[0];

    lvl.ds_factor = 0.25;

    lvl.fixed_imgs_to_use.resize(num_cal_views);
    std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    vout << "    setting up sim metrics..." << std::endl;
    lvl.sim_metrics.reserve(num_cal_views);

    for (size_type view_idx = 0; view_idx < num_cal_views; ++view_idx)
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

      auto pelvis_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      pelvis_guess->vol_idx = 0;

      lvl_regi_coarse.init_mov_vol_poses = { pelvis_guess };

      lvl_regi_coarse.ref_frames = { 0 }; // This refers to pelvis (moving)
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

    if (kSAVE_REGI_DEBUG )
    {
      // Create Debug Proj H5 File
      const std::string proj_data_h5_path = output_path + "/pelvis_multiview_proj_data" + pel_xray_ID + ".h5";
      vout << "creating H5 proj data file for img" + pel_xray_ID + "..." << std::endl;
      H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
      WriteProjDataH5(regi.fixed_proj_data, &h5);

      vout << "  setting regi debug info..." << std::endl;

      DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
      debug_vol_path.vol_path = spec_vol_path;

      if (use_seg)
      {
        debug_vol_path.label_vol_path = spec_seg_path;
        debug_vol_path.labels_used    = { pelvis_label };
      }

      regi.debug_info->vols = { debug_vol_path };

      DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
      debug_proj_path.path = proj_data_h5_path;
      // debug_proj_path.projs_used = { view_idx };

      regi.debug_info->fixed_projs = debug_proj_path;

      regi.debug_info->proj_pre_proc_info = proj_pre_proc.params;

      regi.debug_info->regi_names = { { "Multiview Pelvis" } };
    }

    vout << std::endl << "Multi view pelvis registration ..." << std::endl;
    regi.run();
    regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_pelvis_multiview" + pel_xray_ID + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);

      FrameTransform pelvis_mv_regi_xform = regi.cur_cam_to_vols[0];

      vout << "saving regi transformations ..." << std::endl;
      WriteITKAffineTransform(output_path + "/pelvis_mv_regi_xform.h5", pelvis_mv_regi_xform);
    }
  }
  return 0;
}
