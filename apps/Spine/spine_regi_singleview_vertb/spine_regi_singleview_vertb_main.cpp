
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
  const std::string meta_data_path          = po.pos_args()[1];  // 3D spine landmarks path
  const std::string exp_list_path           = po.pos_args()[2];  // Experiment list file path
  const std::string dicom_path              = po.pos_args()[3];  // Dicom image path
  const std::string output_path             = po.pos_args()[4];  // Output path

  const std::string spinevol_path = meta_data_path + "/Spine21-2512_CT_crop.nrrd";
  const std::string spineseg_path = meta_data_path + "/Spine21-2512_seg_crop.nrrd";
  const std::string sacrumseg_path = meta_data_path + "/Spine21-2512_sacrum_seg_crop.nrrd";
  const std::string spine_3d_fcsv_path = meta_data_path + "/Spine_3D_landmarks.fcsv";

  std::cout << "reading spine anatomical landmarks from FCSV file..." << std::endl;
  std::cout << spine_3d_fcsv_path << std::endl;
  auto spine_3d_fcsv = ReadFCSVFileNamePtMap(spine_3d_fcsv_path);
  ConvertRASToLPS(&spine_3d_fcsv);

  const bool use_seg = true;
  auto spine_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spineseg_path);
  
  auto sacrum_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(sacrumseg_path);

  vout << "reading spine volume..." << std::endl; // We only use the needle metal part
  auto spinevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spinevol_path);
  
  {
    spinevol_hu->SetOrigin(spine_seg->GetOrigin());
    spinevol_hu->SetSpacing(spine_seg->GetSpacing());
  }

  vout << "  HU --> Att. ..." << std::endl;
  auto spinevol_att = HUToLinAtt(spinevol_hu.GetPointer());
  
  {
    sacrum_seg->SetOrigin(spinevol_att->GetOrigin());
    sacrum_seg->SetSpacing(spinevol_att->GetSpacing());
  }
  
  const unsigned char vert1_label = 25;
  const unsigned char vert2_label = 24;
  const unsigned char vert3_label = 23;
  const unsigned char vert4_label = 22;
  const unsigned char sacrum_label = 1;

  unsigned char spine_label = 1;
  vout << "extracting vert label volumes..." << std::endl;
  auto vert1_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert1_label, float(0), true);
  
  auto vert2_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert2_label, float(0), true);
  
  auto vert3_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert3_label, float(0), true);
  
  auto vert4_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), spine_seg.GetPointer(), vert4_label, float(0), true);
  
  auto sacrum_vol = ApplyMaskToITKImage(spinevol_att.GetPointer(), sacrum_seg.GetPointer(), sacrum_label, float(0), true);

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

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert1_singleview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert1-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert1 center" << std::endl;
    }

    vert1_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert1_singleview_regi_ref_frame->vol_idx = 0;
    vert1_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert1_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert1_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert2_singleview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert2-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert2 center" << std::endl;
    }

    vert2_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert2_singleview_regi_ref_frame->vol_idx = 1;
    vert2_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert2_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert2_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert3_singleview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert3-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert3 center" << std::endl;
    }

    vert3_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert3_singleview_regi_ref_frame->vol_idx = 2;
    vert3_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert3_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert3_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert4_singleview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert4-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND vert4 center" << std::endl;
    }

    vert4_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    vert4_singleview_regi_ref_frame->vol_idx = 3;
    vert4_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    vert4_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    vert4_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
  }
  
  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> sacrum_singleview_regi_ref_frame;
  {
    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("sacrum-cen");
    Pt3 rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND sacrum center" << std::endl;
    }

    sacrum_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    sacrum_singleview_regi_ref_frame->vol_idx = 4;
    sacrum_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = rotcenter[0];
    sacrum_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = rotcenter[1];
    sacrum_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = rotcenter[2];
  }

  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  bool is_first_view = true;

  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID                = exp_ID_list[idx];
    const std::string spinebb_2d_fcsv_path  = landmark2d_root_path + "/spine" + exp_ID + ".fcsv";
    const std::string img_path              = dicom_path + "/" + exp_ID;

    std::cout << "Running..." << exp_ID << std::endl;
    auto spinebb_2d_fcsv = ReadFCSVFileNamePtMap(spinebb_2d_fcsv_path);
    ConvertRASToLPS(&spinebb_2d_fcsv);

    xregASSERT(spinebb_2d_fcsv.size() > 3);

    ProjPreProc proj_pre_proc;
    proj_pre_proc.input_projs.resize(1);

    std::vector<CIOSFusionDICOMInfo> spinecios_metas(1);
    {
      std::tie(proj_pre_proc.input_projs[0].img, spinecios_metas[0]) =
                                                      ReadCIOSFusionDICOMFloat(img_path);
      proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(spinecios_metas[0], true);
    }

    {
      UpdateLandmarkMapForCIOSFusion(spinecios_metas[0], &spinebb_2d_fcsv);

      auto& spineproj_lands = proj_pre_proc.input_projs[0].landmarks;
      spineproj_lands.reserve(spinebb_2d_fcsv.size());

      for (const auto& fcsv_kv : spinebb_2d_fcsv)
      {
        spineproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
      }
    }

    proj_pre_proc();
    auto& projs_to_regi = proj_pre_proc.output_projs;

    //FrameTransform init_cam_to_spine = EstCamToWorldBruteForcePOSITCMAESRefine(default_cam, spineproj_lands, spinebb_3d_fcsv);
    FrameTransform init_cam_to_spine = PnPPOSITAndReprojCMAES(projs_to_regi[0].cam, spine_3d_fcsv, projs_to_regi[0].landmarks);

    WriteITKAffineTransform(output_path + "/vertb_singleview_init_xform" + exp_ID + ".h5", init_cam_to_spine);

    MultiLevelMultiObjRegi regi;

    regi.set_debug_output_stream(vout, verbose);
    regi.set_save_debug_info(kSAVE_REGI_DEBUG);
    regi.vols = { vert1_vol, vert2_vol, vert3_vol, vert4_vol, sacrum_vol };
    regi.vol_names = { "vert1", "vert2", "vert3", "vert4", "sacrum" };

    regi.ref_frames = { vert1_singleview_regi_ref_frame, vert2_singleview_regi_ref_frame,
      vert3_singleview_regi_ref_frame, vert4_singleview_regi_ref_frame,
      sacrum_singleview_regi_ref_frame };

    const size_type view_idx = 0;

    regi.fixed_proj_data = proj_pre_proc.output_projs;

    regi.levels.resize(1);

    regi.init_cam_to_vols = { init_cam_to_spine, init_cam_to_spine, init_cam_to_spine, init_cam_to_spine, init_cam_to_spine };

    vert1_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;
    vert2_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;
    vert3_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;
    vert4_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;
    sacrum_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;

    auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();

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
      lvl_regi_coarse.mov_vols = { 0, 1, 2, 3, 4 };
      lvl_regi_coarse.static_vols = { };

      auto vert1_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert1_guess->vol_idx = 0;
      
      auto vert2_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert2_guess->vol_idx = 1;
      
      auto vert3_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert3_guess->vol_idx = 2;
      
      auto vert4_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      vert4_guess->vol_idx = 3;
      
      auto sacrum_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      sacrum_guess->vol_idx = 4;
      
      lvl_regi_coarse.init_mov_vol_poses = { vert1_guess, vert2_guess, vert3_guess, vert4_guess, sacrum_guess };
      lvl_regi_coarse.ref_frames = { 0, 1, 2, 3, 4 };
      {
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

        pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                       std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                       std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                       std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD),
                                       std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };

        pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20),
                                       std::make_shared<FoldNormDist>(20, 20),
                                       std::make_shared<FoldNormDist>(20, 20),
                                       std::make_shared<FoldNormDist>(20, 20),
                                       std::make_shared<FoldNormDist>(20, 20) };

        cmaes_regi->set_pop_size(150);
        cmaes_regi->set_sigma({ 3 * kDEG2RAD, 3 * kDEG2RAD, 3 * kDEG2RAD, 5, 5, 10,
                                3 * kDEG2RAD, 3 * kDEG2RAD, 3 * kDEG2RAD, 5, 5, 10,
                                3 * kDEG2RAD, 3 * kDEG2RAD, 3 * kDEG2RAD, 5, 5, 10,
                                3 * kDEG2RAD, 3 * kDEG2RAD, 3 * kDEG2RAD, 5, 5, 10,
                                3 * kDEG2RAD, 3 * kDEG2RAD, 3 * kDEG2RAD, 5, 5, 10  });

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi_coarse.regi = cmaes_regi;
      }
    }

    if (kSAVE_REGI_DEBUG )
    {
      // Create Debug Proj H5 File
      const std::string proj_data_h5_path = output_path + "/vertb_singleview_proj_data" + exp_ID + ".h5";
      vout << "creating H5 proj data file for img" + exp_ID + "..." << std::endl;
      H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
      WriteProjDataH5(regi.fixed_proj_data, &h5);

      vout << "  setting regi debug info..." << std::endl;

      DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
      debug_vol_path.vol_path = spinevol_path;

      if (use_seg)
      {
        debug_vol_path.label_vol_path = spineseg_path;
        debug_vol_path.labels_used    = { spine_label };
      }

      regi.debug_info->vols = { debug_vol_path };

      DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
      debug_proj_path.path = proj_data_h5_path;
      // debug_proj_path.projs_used = { view_idx };

      regi.debug_info->fixed_projs = debug_proj_path;

      regi.debug_info->proj_pre_proc_info = proj_pre_proc.params;

      regi.debug_info->regi_names = { { "Singleview Spine" + exp_ID } };
    }

    vout << std::endl << "Single view spine registration ..." << std::endl;
    regi.run();
    regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_vertb" + exp_ID + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi.debug_info, dst_debug_path);
    }

    WriteITKAffineTransform(output_path + "/vert1_regi_xform.h5", regi.cur_cam_to_vols[0]);
    WriteITKAffineTransform(output_path + "/vert2_regi_xform.h5", regi.cur_cam_to_vols[1]);
    WriteITKAffineTransform(output_path + "/vert3_regi_xform.h5", regi.cur_cam_to_vols[2]);
    WriteITKAffineTransform(output_path + "/vert4_regi_xform.h5", regi.cur_cam_to_vols[3]);
    WriteITKAffineTransform(output_path + "/sacrum_regi_xform.h5", regi.cur_cam_to_vols[4]);

    {
      for(size_type vol_idx=0; vol_idx<5; ++vol_idx)
      {
        auto ray_caster = LineIntRayCasterFromProgOpts(po);
        ray_caster->set_camera_model(default_cam);
        ray_caster->use_proj_store_replace_method();
        ray_caster->set_volume(regi.vols[vol_idx]);
        ray_caster->set_num_projs(1);
        ray_caster->allocate_resources();
        ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[vol_idx];
        ray_caster->compute(0);

        WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/vert_reproj" + exp_ID + "_vert" + fmt::format("{:02d}", vol_idx) + ".png");
      }
      WriteITKImageRemap8bpp(proj_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real" + exp_ID + ".png");
    }

    {
    	const std::string spine_3d_bb_fcsv_path = meta_data_path + "/phantom_bbs_corrected.fcsv";

    	LandMap3 reproj_bbs_fcsv;

	    std::cout << "reading spine BB landmarks from FCSV file..." << std::endl;
	    auto spine_3d_bb_fcsv = ReadFCSVFileNamePtMap(spine_3d_bb_fcsv_path);
	    ConvertRASToLPS(&spine_3d_bb_fcsv);

	    for( const auto& n : spine_3d_bb_fcsv )
	    {
	      auto reproj_bb = default_cam.phys_pt_to_ind_pt(Pt3(regi.cur_cam_to_vols[0].inverse() *  n.second));
	      reproj_bb[0] = 0.194 * (reproj_bb[0] - 1536);
	      reproj_bb[1] = 0.194 * (reproj_bb[1] - 1536);
	      reproj_bb[2] = 0;
	      std::pair<std::string, Pt3> ld2_3D(n.first, reproj_bb);
	      reproj_bbs_fcsv.insert(ld2_3D);
	    }

	    WriteFCSVFileFromNamePtMap(output_path + "/reproj_bb" + exp_ID + ".fcsv", reproj_bbs_fcsv);
    }
  }
  return 0;
}
