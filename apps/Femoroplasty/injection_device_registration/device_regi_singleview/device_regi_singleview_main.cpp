
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

LandMap3 Flip2DAnnotation(LandMap3 devicebb_2d_fcsv)
{
  LandMap3 devicebb_2d_fcsv_new;
  for (const auto& fcsv_kv : devicebb_2d_fcsv)
  {
    if(fcsv_kv.first == "LeftScrewHead")
      devicebb_2d_fcsv_new.emplace("RightScrewHead", fcsv_kv.second);
    else if(fcsv_kv.first == "RightScrewHead")
      devicebb_2d_fcsv_new.emplace("LeftScrewHead", fcsv_kv.second);

    if(fcsv_kv.first == "LeftScrewTip")
      devicebb_2d_fcsv_new.emplace("RightScrewTip", fcsv_kv.second);
    else if(fcsv_kv.first == "RightScrewTip")
      devicebb_2d_fcsv_new.emplace("LeftScrewTip", fcsv_kv.second);
  }

  return devicebb_2d_fcsv_new;
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Single-view Device Registration");
  po.set_arg_usage("<Device 2D landmark annotation ROOT path> <Device 3D landmark annotation FILE path> <Device 3D BB FILE path> <Image ID list txt file path> <Image DICOM ROOT path> <Output folder path>");
  po.set_min_num_pos_args(6);

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
  const std::string device_3d_fcsv_path     = po.pos_args()[1];  // 3D device landmarks path
  const std::string device_3d_bb_fcsv_path  = po.pos_args()[2];  // 3D device BB annotation path
  const std::string exp_list_path           = po.pos_args()[3];  // Experiment list file path
  const std::string dicom_path              = po.pos_args()[4];  // Dicom image path
  const std::string output_path             = po.pos_args()[5];  // Output path

  std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

  const std::string devicevol_path = "/home/cong/Research/Spine/CadaverNeedleInjection/Device_crop.nii.gz";
  const std::string deviceseg_path = "/home/cong/Research/Spine/CadaverNeedleInjection/Device_crop_seg.nii.gz";

  const bool use_seg = true;
  auto device_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(deviceseg_path);

  vout << "reading device volume..." << std::endl; // We only use the needle metal part
  auto devicevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(devicevol_path);

  vout << "  HU --> Att. ..." << std::endl;

  auto devicevol_att = HUToLinAtt(devicevol_hu.GetPointer());

  unsigned char device_label = 1;

  auto device_vol = ApplyMaskToITKImage(devicevol_att.GetPointer(), device_seg.GetPointer(), device_label, float(0), true);

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

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> device_singleview_regi_ref_frame;

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
      vout << "ERROR: NOT FOUND DRILL ROT CENTER" << std::endl;
    }

    device_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    device_singleview_regi_ref_frame->vol_idx = 0;
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = device_rotcenter[0];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = device_rotcenter[1];
    device_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = device_rotcenter[2];
  }

  if(lineNumber!=exp_ID_list.size()) throw std::runtime_error("Exp ID list size mismatch!!!");

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  bool is_first_view = true;

  for(int idx=0; idx<lineNumber; ++idx)
  {
    for(size_type flipidx=0; flipidx<2; ++flipidx)
    {
      const std::string exp_ID                = exp_ID_list[idx];
      const std::string devicebb_2d_fcsv_path  = landmark2d_root_path + "/" + exp_ID + ".fcsv";
      const std::string img_path              = dicom_path + "/" + exp_ID;

      std::cout << "Running..." << exp_ID << std::endl;
      auto devicebb_2d_fcsv = ReadFCSVFileNamePtMap(devicebb_2d_fcsv_path);
      ConvertRASToLPS(&devicebb_2d_fcsv);

      xregASSERT(devicebb_2d_fcsv.size() > 3);

      if(flipidx == 1)
        devicebb_2d_fcsv = Flip2DAnnotation(devicebb_2d_fcsv);

      LandMap2 deviceproj_lands;
      /*
      for (const auto& fcsv_kv : bb_fcsv)
      {
        deviceproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
      }
       */

      ProjPreProc proj_pre_proc;
      proj_pre_proc.input_projs.resize(1);

      std::vector<CIOSFusionDICOMInfo> devicecios_metas(1);
      {
        std::tie(proj_pre_proc.input_projs[0].img, devicecios_metas[0]) =
                                                        ReadCIOSFusionDICOMFloat(img_path);
        proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(devicecios_metas[0], true);
      }

      {
        UpdateLandmarkMapForCIOSFusion(devicecios_metas[0], &devicebb_2d_fcsv);

        auto& deviceproj_lands = proj_pre_proc.input_projs[0].landmarks;
        deviceproj_lands.reserve(devicebb_2d_fcsv.size());

        for (const auto& fcsv_kv : devicebb_2d_fcsv)
        {
          deviceproj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
        }
      }

      proj_pre_proc();
      auto& projs_to_regi = proj_pre_proc.output_projs;

      FrameTransform init_cam_to_device = PnPPOSITAndReprojCMAES(projs_to_regi[0].cam, device_3d_fcsv, projs_to_regi[0].landmarks);

      // WriteITKAffineTransform(output_path + "/device_init_xform" + exp_ID + ".h5", init_cam_to_device);

      MultiLevelMultiObjRegi regi;

      regi.set_debug_output_stream(vout, verbose);
      regi.set_save_debug_info(kSAVE_REGI_DEBUG);
      regi.vols = { device_vol };
      regi.vol_names = { "Device"};

      regi.ref_frames = { device_singleview_regi_ref_frame };

      const size_type view_idx = 0;

      regi.fixed_proj_data = proj_pre_proc.output_projs;

      regi.levels.resize(1);

      regi.init_cam_to_vols = { init_cam_to_device };

      device_singleview_regi_ref_frame->cam_extrins = regi.fixed_proj_data[view_idx].cam.extrins;

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

          cmaes_regi->set_pop_size(20);
          cmaes_regi->set_sigma({ 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 25 });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_coarse.regi = cmaes_regi;
        }
        vout << std::endl << "First view spine coarse registration ..." << std::endl;
        regi.run();
        regi.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

      }

      // WriteITKAffineTransform(output_path + "/device_regi_xform" + exp_ID + ".h5", regi.cur_cam_to_vols[0]);

      {
        auto ray_caster = LineIntRayCasterFromProgOpts(po);
        ray_caster->set_camera_model(default_cam);
        ray_caster->use_proj_store_replace_method();
        ray_caster->set_volume(device_vol);
        ray_caster->set_num_projs(1);
        ray_caster->allocate_resources();
        ray_caster->xform_cam_to_itk_phys(0) = regi.cur_cam_to_vols[0];
        ray_caster->compute(0);

        if(flipidx == 0)
        {
          WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/device_reproj" + exp_ID + ".png");
          WriteITKImageRemap8bpp(proj_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real" + exp_ID + ".png");
        }
        else{
          WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/device_reproj" + exp_ID + "_flip.png");
          WriteITKImageRemap8bpp(proj_pre_proc.input_projs[0].img.GetPointer(), output_path + "/real" + exp_ID + "_flip.png");
        }

      }

      LandMap3 reproj_bbs_fcsv;

      std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
      auto device_3d_bb_fcsv = ReadFCSVFileNamePtMap(device_3d_bb_fcsv_path);
      ConvertRASToLPS(&device_3d_bb_fcsv);

      for( const auto& n : device_3d_bb_fcsv )
      {
        auto reproj_bb = default_cam.phys_pt_to_ind_pt(Pt3(regi.cur_cam_to_vols[0].inverse() *  n.second));
        reproj_bb[0] = 0.194 * (reproj_bb[0] - 1536);
        reproj_bb[1] = 0.194 * (reproj_bb[1] - 1536);
        reproj_bb[2] = 0;
        std::pair<std::string, Pt3> ld2_3D(n.first, reproj_bb);
        reproj_bbs_fcsv.insert(ld2_3D);
      }

      if(flipidx == 0)
        WriteFCSVFileFromNamePtMap(output_path + "/reproj_bb" + exp_ID + ".fcsv", reproj_bbs_fcsv);
      else
        WriteFCSVFileFromNamePtMap(output_path + "/reproj_bb" + exp_ID + "_flip.fcsv", reproj_bbs_fcsv);
    }
  }
  return 0;
}
