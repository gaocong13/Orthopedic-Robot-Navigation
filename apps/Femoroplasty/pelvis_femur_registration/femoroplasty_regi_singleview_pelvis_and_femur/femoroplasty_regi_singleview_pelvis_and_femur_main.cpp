
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

  const std::string landmark2d_root_path    = po.pos_args()[0];  // 2D Landmark root path
  const std::string meta_data_path          = po.pos_args()[1];  // 3D femur landmarks path
  const std::string exp_list_path           = po.pos_args()[2];  // Experiment list file path
  const std::string dicom_path              = po.pos_args()[3];  // Dicom image path
  const std::string output_path             = po.pos_args()[4];  // Output path

  const std::string spec_vol_path = meta_data_path + "/Spec21-2512_CT_flipped.nii.gz";
  const std::string spec_seg_path = meta_data_path + "/Spec21-2512_Seg_flipped.nii.gz";
  const std::string pelvis_3d_fcsv_path = meta_data_path + "/pelvis_anatomical_landmark.fcsv";

  const size_type num_views = 1; // This is single-view registration

  const bool use_seg = true;
  auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();
  auto so3_vars = std::make_shared<SO3OptVarsLieAlg>();

  std::cout << "reading pelvis anatomical landmarks from FCSV file..." << std::endl;
  auto pelvis_3d_fcsv = ReadFCSVFileNamePtMap(pelvis_3d_fcsv_path);
  ConvertRASToLPS(&pelvis_3d_fcsv);

  vout << "reading femur CT volume..." << std::endl;
  auto vol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spec_vol_path);

  vout << "  HU --> Att. ..." << std::endl;
  auto vol_att = HUToLinAtt(vol_hu.GetPointer());

  auto vol_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spec_seg_path);

  /*
  {
    femurvol_hu->SetOrigin(femur_seg->GetOrigin());
    femurvol_hu->SetSpacing(femur_seg->GetSpacing());
  }
  */

  unsigned char pelvis_label = 3;
  unsigned char femur_label  = 1;

  vout << "cropping intensity volume tightly around labels:"
       << "\n  Pelvis: " << static_cast<int>(pelvis_label)
       << "\n   Femur: " << static_cast<int>(femur_label)
       << std::endl;

  auto ct_vols = MakeVolListFromVolAndLabels(vol_att.GetPointer(), vol_seg.GetPointer(),
                                             { pelvis_label, femur_label }, 0);

  vout << "extracting pelvis att. volume..." << std::endl;
  auto pelvis_vol = ApplyMaskToITKImage(vol_att.GetPointer(), vol_seg.GetPointer(), pelvis_label, float(0), true);

  vout << "extracting femur att. volume..." << std::endl;
  auto femur_vol = ApplyMaskToITKImage(vol_att.GetPointer(), vol_seg.GetPointer(), femur_label, float(0), true);

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> pelvis_singleview_regi_ref_frame;
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
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> femur_singleview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto femur_fcsv_rotc = pelvis_3d_fcsv.find("FH-r");
    Pt3 rotcenter;

    if (femur_fcsv_rotc != pelvis_3d_fcsv.end()){
      rotcenter = femur_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert2 center" << std::endl;
    }

    femur_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    femur_singleview_regi_ref_frame->vol_idx = 1;
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
  }

  // Read X-ray Image

  std::vector<std::string> exp_ID_list;
  int lineNumber = 0;
  /* Read exp ID list from file */
  {
    std::ifstream expIDFile(exp_list_path);
    // Make sure the file is open
    if(!expIDFile.is_open()) throw std::runtime_error("Could not open exp ID file");

    std::string line, csvItem;

    while(std::getline(expIDFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          exp_ID_list.push_back(csvItem);
      }
      lineNumber++;
    }
  }

  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  if(lineNumber!=num_views) throw std::runtime_error("More than One image parsed!!!");


  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  bool is_first_view = true;

  for(int idx=0; idx<num_views; ++idx)
  {
    const std::string exp_ID                        = exp_ID_list[idx];
    const std::string pelvis_landmark_2d_fcsv_path  = landmark2d_root_path + "/pelvis" + exp_ID + ".fcsv";
    const std::string img_path                      = dicom_path + "/" + exp_ID;

    std::cout << "Running..." << exp_ID << std::endl;
    auto pelvis_landmark_2d_fcsv = ReadFCSVFileNamePtMap(pelvis_landmark_2d_fcsv_path);
    ConvertRASToLPS(&pelvis_landmark_2d_fcsv);

    xregASSERT(pelvis_landmark_2d_fcsv.size() > 3);

    ProjPreProc proj_pre_proc;
    proj_pre_proc.input_projs.resize(1);

    std::vector<CIOSFusionDICOMInfo> cios_metas(1);
    {
      std::tie(proj_pre_proc.input_projs[0].img, cios_metas[0]) =
                                                      ReadCIOSFusionDICOMFloat(img_path);
      proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(cios_metas[0], true);
    }

    {
      UpdateLandmarkMapForCIOSFusion(cios_metas[0], &pelvis_landmark_2d_fcsv);

      auto& pelvis_proj_lands = proj_pre_proc.input_projs[0].landmarks;
      pelvis_proj_lands.reserve(pelvis_landmark_2d_fcsv.size());

      for (const auto& fcsv_kv : pelvis_landmark_2d_fcsv)
      {
        pelvis_proj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
      }
    }

    proj_pre_proc();
    auto& projs_to_regi = proj_pre_proc.output_projs;

    vout << "running initialization..." << std::endl;

    FrameTransform init_cam_to_pelvis = PnPPOSITAndReprojCMAES(projs_to_regi[0].cam, pelvis_3d_fcsv, projs_to_regi[0].landmarks);
    FrameTransform init_cam_to_femur = init_cam_to_pelvis;

    WriteITKAffineTransform(output_path + "/pelvis_singleview_init_xform" + exp_ID + ".h5", init_cam_to_pelvis);
    WriteITKAffineTransform(output_path + "/femur_singleview_init_xform" + exp_ID + ".h5",  init_cam_to_femur);

    const size_type view_idx = 0;

    MultiLevelMultiObjRegi regi;

    regi.set_debug_output_stream(vout, verbose);
    regi.set_save_debug_info(kSAVE_REGI_DEBUG);
    regi.vols = { pelvis_vol, femur_vol };
    regi.vol_names = { "pelvis", "femur" };

    regi.ref_frames = { pelvis_singleview_regi_ref_frame, femur_singleview_regi_ref_frame };

    regi.fixed_proj_data = proj_pre_proc.output_projs;

    pelvis_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;
    femur_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;

    regi.levels.resize(1);

    // Pelvis Registration First
    {
      regi.init_cam_to_vols = { init_cam_to_pelvis, init_cam_to_femur };

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

        auto pelvis_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        pelvis_guess->vol_idx = 0;

        auto femur_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        femur_guess->vol_idx = 1;

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
        const std::string proj_data_h5_path = output_path + "/pelvis_singleview_proj_data" + exp_ID + ".h5";
        vout << "creating H5 proj data file for img" + exp_ID + "..." << std::endl;
        H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
        WriteProjDataH5(regi.fixed_proj_data, &h5);

        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
        debug_vol_path.vol_path = spec_vol_path;

        if (use_seg)
        {
          debug_vol_path.label_vol_path = spec_seg_path;
          debug_vol_path.labels_used    = { pelvis_label, femur_label };
        }

        regi.debug_info->vols = { debug_vol_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = proj_data_h5_path;
        // debug_proj_path.projs_used = { view_idx };

        regi.debug_info->fixed_projs = debug_proj_path;

        regi.debug_info->proj_pre_proc_info = proj_pre_proc.params;

        regi.debug_info->regi_names = { { "Singleview Pelvis Femur" + exp_ID } };
      }

      vout << std::endl << "Single view pelvis registration ..." << std::endl;
      regi.run();
      regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

      if (kSAVE_REGI_DEBUG)
      {
        vout << "writing debug info to disk..." << std::endl;
        const std::string dst_debug_path = output_path + "/debug_pelvis" + exp_ID + ".h5";
        WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
      }

      WriteITKAffineTransform(output_path + "/pelvis_regi_xform.h5", regi.cur_cam_to_vols[0]);

      {

        auto ray_caster = LineIntRayCasterFromProgOpts(po);
        ray_caster->set_camera_model(default_cam);
        ray_caster->use_proj_store_replace_method();
        ray_caster->set_volume(regi.vols[0]);
        ray_caster->set_num_projs(1);
        ray_caster->allocate_resources();
        ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[0];
        ray_caster->compute(0);

        WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/pelvis_reproj" + exp_ID + ".png");
        WriteITKImageRemap8bpp(proj_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real" + exp_ID + ".png");
      }
    }

    // Femur Registration Next
    {
      regi.init_cam_to_vols = { regi.cur_cam_to_vols[0], regi.cur_cam_to_vols[1] };

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

      lvl.regis.resize(2);

      {
        auto& lvl_regi_coarse = lvl.regis[0];
        lvl_regi_coarse.mov_vols = { 1 };
        lvl_regi_coarse.static_vols = { 0 };

        auto pelvis_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        pelvis_guess->vol_idx = 0;

        auto femur_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        femur_guess->vol_idx = 1;

        lvl_regi_coarse.init_mov_vol_poses = { femur_guess };
        lvl_regi_coarse.static_vol_poses = { pelvis_guess };

        lvl_regi_coarse.ref_frames = { 1 }; // This refers to pelvis (moving)
        {
          auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
          cmaes_regi->set_opt_vars(so3_vars);
          cmaes_regi->set_opt_x_tol(0.01);
          cmaes_regi->set_opt_obj_fn_tol(0.01);

          auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(20 * kDEG2RAD, 20 * kDEG2RAD) };
          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20) };

          cmaes_regi->set_pop_size(150);
          cmaes_regi->set_sigma({ 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_coarse.regi = cmaes_regi;
        }
      }

      {
        auto& lvl_regi_fine = lvl.regis[1];
        lvl_regi_fine.mov_vols = { 1 };
        lvl_regi_fine.static_vols = { 0 };

        auto pelvis_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        pelvis_guess->vol_idx = 0;

        auto femur_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        femur_guess->vol_idx = 1;

        lvl_regi_fine.init_mov_vol_poses = { femur_guess };
        lvl_regi_fine.static_vol_poses = { pelvis_guess };

        lvl_regi_fine.ref_frames = { 1 }; // This refers to pelvis (moving)
        {
          auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
          cmaes_regi->set_opt_vars(se3_vars);
          cmaes_regi->set_opt_x_tol(0.01);
          cmaes_regi->set_opt_obj_fn_tol(0.01);

          auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(5 * kDEG2RAD, 5 * kDEG2RAD) };

          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(30, 30) };

          cmaes_regi->set_pop_size(150);
          cmaes_regi->set_sigma({ 1 * kDEG2RAD, 1 * kDEG2RAD, 1 * kDEG2RAD, 5, 5, 10 });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_fine.regi = cmaes_regi;
        }
      }

      if (kSAVE_REGI_DEBUG )
      {
        // Create Debug Proj H5 File
        const std::string proj_data_h5_path = output_path + "/femur_singleview_proj_data" + exp_ID + ".h5";
        vout << "creating H5 proj data file for img" + exp_ID + "..." << std::endl;
        H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
        WriteProjDataH5(regi.fixed_proj_data, &h5);

        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
        debug_vol_path.vol_path = spec_vol_path;

        if (use_seg)
        {
          debug_vol_path.label_vol_path = spec_seg_path;
          debug_vol_path.labels_used    = { pelvis_label, femur_label };
        }

        regi.debug_info->vols = { debug_vol_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = proj_data_h5_path;
        // debug_proj_path.projs_used = { view_idx };

        regi.debug_info->fixed_projs = debug_proj_path;

        regi.debug_info->proj_pre_proc_info = proj_pre_proc.params;

        regi.debug_info->regi_names = { { "Singleview Femur Coarse" + exp_ID, "Singleview Femur Fine" + exp_ID } };
      }

      vout << std::endl << "Single view pelvis registration ..." << std::endl;
      regi.run();
      regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

      if (kSAVE_REGI_DEBUG)
      {
        vout << "writing debug info to disk..." << std::endl;
        const std::string dst_debug_path = output_path + "/debug_femur" + exp_ID + ".h5";
        WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
      }

      WriteITKAffineTransform(output_path + "/femur_regi_xform.h5", regi.cur_cam_to_vols[1]);

      {
        auto ray_caster = LineIntRayCasterFromProgOpts(po);
        ray_caster->set_camera_model(default_cam);
        ray_caster->use_proj_store_replace_method();
        ray_caster->set_volume(regi.vols[1]);
        ray_caster->set_num_projs(1);
        ray_caster->allocate_resources();
        ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[1];
        ray_caster->compute(0);

        WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/femur_reproj" + exp_ID + ".png");
      }
    }
  }
  return 0;
}
