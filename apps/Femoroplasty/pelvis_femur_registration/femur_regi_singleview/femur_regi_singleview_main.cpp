
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
  const std::string femur_3d_fcsv_path      = po.pos_args()[1];  // 3D femur landmarks path
  const std::string exp_list_path           = po.pos_args()[2];  // Experiment list file path
  const std::string dicom_path              = po.pos_args()[3];  // Dicom image path
  const std::string output_path             = po.pos_args()[4];  // Output path

  std::cout << "reading femur BB landmarks from FCSV file..." << std::endl;
  auto femur_3d_fcsv = ReadFCSVFileNamePtMap(femur_3d_fcsv_path);
  ConvertRASToLPS(&femur_3d_fcsv);

  const std::string femurvol_path = "/home/cong/Research/Femoroplasty/Phantom_Injection/meta_data/Femur_CT_crop.nii.gz";
  const std::string femurseg_path = "/home/cong/Research/Femoroplasty/Phantom_Injection/meta_data/Femur_seg_crop.nii.gz";

  const bool use_seg = true;
  auto femur_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(femurseg_path);

  vout << "reading femur volume..." << std::endl; // We only use the needle metal part
  auto femurvol_hu = ReadITKImageFromDisk<RayCaster::Vol>(femurvol_path);

  vout << "  HU --> Att. ..." << std::endl;
  auto femurvol_att = HUToLinAtt(femurvol_hu.GetPointer());

  unsigned char femur_label = 1;
  auto femur_vol = ApplyMaskToITKImage(femurvol_att.GetPointer(), femur_seg.GetPointer(), femur_label, float(0), true);

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

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> femur_singleview_regi_ref_frame;

  {
    vout << "setting up femur ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal femur volume center point as the origin

    itk::ContinuousIndex<double,3> center_idx;

    auto femur_fcsv_rotc = femur_3d_fcsv.find("FH");
    Pt3 femur_rotcenter;

    if (femur_fcsv_rotc != femur_3d_fcsv.end()){
      femur_rotcenter = femur_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND femur femur head center" << std::endl;
    }

    femur_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    femur_singleview_regi_ref_frame->vol_idx = 0;
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = femur_rotcenter[0];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = femur_rotcenter[1];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = femur_rotcenter[2];
  }

  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  bool is_first_view = true;

  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID                = exp_ID_list[idx];
    const std::string femurbb_2d_fcsv_path  = landmark2d_root_path + "/Femur" + exp_ID + ".fcsv";
    const std::string img_path              = dicom_path + "/" + exp_ID;

    std::cout << "Running..." << exp_ID << std::endl;
    auto femurbb_2d_fcsv = ReadFCSVFileNamePtMap(femurbb_2d_fcsv_path);
    ConvertRASToLPS(&femurbb_2d_fcsv);

    xregASSERT(femurbb_2d_fcsv.size() > 3);

    ProjPreProc proj_pre_proc;
    proj_pre_proc.input_projs.resize(1);

    std::vector<CIOSFusionDICOMInfo> femurcios_metas(1);
    {
      std::tie(proj_pre_proc.input_projs[0].img, femurcios_metas[0]) =
                                                      ReadCIOSFusionDICOMFloat(img_path);
      proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(femurcios_metas[0], true);
    }

    {
      UpdateLandmarkMapForCIOSFusion(femurcios_metas[0], &femurbb_2d_fcsv);

      auto& femurproj_lands = proj_pre_proc.input_projs[0].landmarks;
      femurproj_lands.reserve(femurbb_2d_fcsv.size());

      for (const auto& fcsv_kv : femurbb_2d_fcsv)
      {
        femurproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
      }
    }

    proj_pre_proc();
    auto& projs_to_regi = proj_pre_proc.output_projs;
    //FrameTransform init_cam_to_femur = EstCamToWorldBruteForcePOSITCMAESRefine(default_cam, femurproj_lands, femurbb_3d_fcsv);
    FrameTransform init_cam_to_femur = PnPPOSITAndReprojCMAES(projs_to_regi[0].cam, femur_3d_fcsv, projs_to_regi[0].landmarks);
/*
    {
      std::vector<SingleProjData> tmp_pd_list(1);
      {
        tmp_pd_list[0].img = projs_to_regi[0].img;
        tmp_pd_list[0].cam = projs_to_regi[0].cam;
      }
      WriteProjData(output_path+"/proj_dr_data" + exp_ID + ".xml", tmp_pd_list);
    }
*/
    WriteITKAffineTransform(output_path + "/femur_init_xform" + exp_ID + ".h5", init_cam_to_femur);

    MultiLevelMultiObjRegi regi;

    regi.set_debug_output_stream(vout, verbose);
    regi.set_save_debug_info(kSAVE_REGI_DEBUG);
    regi.vols = { femur_vol };
    regi.vol_names = { "femur"};

    regi.ref_frames = { femur_singleview_regi_ref_frame };

    const size_type view_idx = 0;

    regi.fixed_proj_data = proj_pre_proc.output_projs;

    regi.levels.resize(1);

    regi.init_cam_to_vols = { init_cam_to_femur };

    femur_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;

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
      lvl_regi_coarse.mov_vols = { 0 };
      lvl_regi_coarse.static_vols = { };

      auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      init_guess->vol_idx = 0;
      lvl_regi_coarse.init_mov_vol_poses = { init_guess };
      lvl_regi_coarse.ref_frames = { 0 };
      {
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();

        pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(2.5 * kDEG2RAD, 2.5 * kDEG2RAD) };
        pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(5, 5) };

        cmaes_regi->set_pop_size(150);
        cmaes_regi->set_sigma({ 30 * kDEG2RAD, 30 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi_coarse.regi = cmaes_regi;
      }
      vout << std::endl << "First view femur registration ..." << std::endl;
      regi.run();
      regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

      if(kSAVE_REGI_DEBUG)
      {
        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
        debug_vol_path.vol_path = femurvol_path;

        if (use_seg)
        {
          debug_vol_path.label_vol_path = femurseg_path;
          debug_vol_path.labels_used    = { femur_label };
        }

        regi.debug_info->vols = { debug_vol_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = img_path;
        debug_proj_path.projs_used = { view_idx };

        regi.debug_info->fixed_projs = debug_proj_path;

        regi.debug_info->proj_pre_proc_info = proj_pre_proc.params;

        regi.debug_info->regi_names = { { "femur" } };

      }
    }

    WriteITKAffineTransform(output_path + "/femur_regi_xform" + exp_ID + ".h5", regi.cur_cam_to_vols[0]);

    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_model(default_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volume(femur_vol);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[0];
      ray_caster->compute(0);

      WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/femur_reproj" + exp_ID + ".png");
      WriteITKImageRemap8bpp(proj_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real" + exp_ID + ".png");
    }
  }
  return 0;
}
