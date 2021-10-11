
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
#include "xregHUToLinAtt.h"
#include "xregImageAddPoissonNoise.h"
#include "xregSampleUtils.h"
#include "xregSampleUniformUnitVecs.h"
#include "xregRotUtils.h"
#include "xregRigidUtils.h"

#include "xregPAODrawBones.h"
#include "xregRigidUtils.h"

#include "bigssMath.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVersor.h"
#include "itkChangeInformationImageFilter.h"

#include "xregRecomposeVertebraes.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = false;

using size_type = std::size_t;
using CamModelList = std::vector<CameraModel>;

using Pt3         = Eigen::Matrix<CoordScalar,3,1>;
using Pt2         = Eigen::Matrix<CoordScalar,2,1>;

FrameTransform Delta_rand_matrix(float rot_mag, float trans_mag)
{
  std::mt19937 rng_eng;
  SeedRNGEngWithRandDev(&rng_eng);
  // for sampling random rotation axes and random translation directions
  UniformOnUnitSphereDist unit_vec_dist(3);

  // for sampling random rotation angles (in APP w/ origin at FH)
  std::uniform_real_distribution<CoordScalar> rot_ang_dist(-rot_mag*kDEG2RAD, rot_mag*kDEG2RAD);

  // for sampling random translation magnitudes (in APP)
  std::uniform_real_distribution<CoordScalar> trans_mag_dist(-trans_mag, trans_mag);

  FrameTransform delta_ref = FrameTransform::Identity();
  {
    // add some noise in the APP coordinate frame
    Pt3 so3 = unit_vec_dist(rng_eng);
    so3 *= rot_ang_dist(rng_eng);

    Eigen::Matrix<CoordScalar,3,3> rot_mat;
    rot_mat = ExpSO3(so3);
    delta_ref.matrix().block(0,0,3,3) = rot_mat;

    Pt3 trans = unit_vec_dist(rng_eng);
    trans *= trans_mag_dist(rng_eng);
    delta_ref.matrix().block(0,3,3,1) = trans;
  }

  return delta_ref;
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

po.set_help("Simulation of Multi-view Spine Registration");
  po.set_arg_usage("< meta data path > < output path >");
  po.set_min_num_pos_args(2);

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

  const std::string meta_data_path          = po.pos_args()[0];  // 3D spine landmarks path
  const std::string output_path             = po.pos_args()[1];  // Output path

  const std::string spinevol_path = meta_data_path + "/Spine21-2512_CT_crop.nrrd";
  const std::string spineseg_path = meta_data_path + "/Spine21-2512_seg_crop.nrrd";
  const std::string sacrumseg_path = meta_data_path + "/Spine21-2512_sacrum_seg_crop.nrrd";
  const std::string spine_gt_xform_path = meta_data_path + "/sacrum_regi_xform.h5";
  const std::string device_gt_xform_view0_path = meta_data_path + "/device_regi_xform01.h5";
  const std::string device_gt_xform_view1_path = meta_data_path + "/device_regi_xform05.h5";
  const std::string device_gt_xform_view2_path = meta_data_path + "/device_regi_xform09.h5";
  const std::string spine_3d_fcsv_path = meta_data_path + "/Spine_3D_landmarks.fcsv";

  const std::string device_3d_fcsv_path    = meta_data_path + "/Device3Dlandmark.fcsv";
  const std::string device_3d_bb_fcsv_path = meta_data_path + "/Device3Dbb.fcsv";
  const std::string devicevol_path         = meta_data_path + "/Device_crop_CT.nii.gz";
  const std::string deviceseg_path         = meta_data_path + "/Device_crop_seg.nii.gz";

  const size_type num_views = 3;

  const float rot_mag_leftright = 5.; // Vertebrae axis Y in degrees
  const float rot_mag_backforth = 5.; // Vertebrae axis X in degrees
  const float rot_mag_tilt = 5.;       // Vertebrae axis Z in degrees
  const float rot_mag_device = 10;     // Device initialization random rotation in degrees;
  const float trans_mag_device = 10;   // Device initialization random translation in mm;
  const float rot_mag_spine = 10;      // Spine initialization random rotation in degrees;
  const float trans_mag_spine = 10;    // Spine initialization random translation in mm;

  vout << "reading spine anatomical landmarks from FCSV file..." << std::endl;
  auto spine_3d_fcsv = ReadFCSVFileNamePtMap(spine_3d_fcsv_path);
  ConvertRASToLPS(&spine_3d_fcsv);

  vout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

  const bool use_seg = true;
  auto spine_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(spineseg_path);

  auto sacrum_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(sacrumseg_path);

  vout << "reading spine volume..." << std::endl; // We only use the needle metal part
  auto spinevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(spinevol_path);

  // Resample Spine Vertebraes Popses
  std::mt19937 rng_eng;
  SeedRNGEngWithRandDev(&rng_eng);
  std::uniform_real_distribution<float> rot_angX_dist(-rot_mag_backforth, rot_mag_backforth);
  std::uniform_real_distribution<float> rot_angY_dist(-rot_mag_leftright, rot_mag_leftright);
  std::uniform_real_distribution<float> rot_angZ_dist(-rot_mag_tilt, rot_mag_tilt);

  {
    spinevol_hu->SetOrigin(spine_seg->GetOrigin());
    spinevol_hu->SetSpacing(spine_seg->GetSpacing());
  }

  vout << "  HU --> Att. ..." << std::endl;
  auto spine_att = HUToLinAtt(spinevol_hu.GetPointer());

  {
    sacrum_seg->SetOrigin(spine_att->GetOrigin());
    sacrum_seg->SetSpacing(spine_att->GetSpacing());
  }

  const unsigned char vert1_label = 25;
  const unsigned char vert2_label = 24;
  const unsigned char vert3_label = 23;
  const unsigned char vert4_label = 22;
  const unsigned char sacrum_label = 1;

  unsigned char spine_label = 1;
  vout << "extracting vert label volumes..." << std::endl;
  auto vert1_att = ApplyMaskToITKImage(spine_att.GetPointer(), spine_seg.GetPointer(), vert1_label, float(0), true);

  auto vert2_att = ApplyMaskToITKImage(spine_att.GetPointer(), spine_seg.GetPointer(), vert2_label, float(0), true);

  auto vert3_att = ApplyMaskToITKImage(spine_att.GetPointer(), spine_seg.GetPointer(), vert3_label, float(0), true);

  auto vert4_att = ApplyMaskToITKImage(spine_att.GetPointer(), spine_seg.GetPointer(), vert4_label, float(0), true);

  auto sacrum_att = ApplyMaskToITKImage(spine_att.GetPointer(), sacrum_seg.GetPointer(), sacrum_label, float(0), true);

  auto device_seg = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(deviceseg_path);

  vout << "reading device volume..." << std::endl; // We only use the needle metal part
  auto devicevol_hu = ReadITKImageFromDisk<RayCaster::Vol>(devicevol_path);

  vout << "  HU --> Att. ..." << std::endl;

  auto devicevol_att = HUToLinAtt(devicevol_hu.GetPointer());

  unsigned char device_label = 1;

  auto device_att = ApplyMaskToITKImage(devicevol_att.GetPointer(), device_seg.GetPointer(), device_label, float(0), true);

  const FrameTransform gt_cam_wrt_spine = ReadITKAffineTransformFromFile(spine_gt_xform_path);
  // TODO: Modify the device_gt_xform_path
  const FrameTransform gt_cam_wrt_device_view0 = ReadITKAffineTransformFromFile(device_gt_xform_view0_path);
  const FrameTransform gt_cam_wrt_device_view1 = ReadITKAffineTransformFromFile(device_gt_xform_view1_path);
  const FrameTransform gt_cam_wrt_device_view2 = ReadITKAffineTransformFromFile(device_gt_xform_view2_path);

  FrameTransformList gt_cam_wrt_device_list = { gt_cam_wrt_device_view0,
                                                gt_cam_wrt_device_view1,
                                                gt_cam_wrt_device_view2 };
  FrameTransformList gt_cam_wrt_spine_list;
  gt_cam_wrt_spine_list.reserve(num_views);
  for (size_type idx = 0; idx < num_views; ++idx)
  {
      gt_cam_wrt_spine_list[idx] = gt_cam_wrt_spine * gt_cam_wrt_device_list[0].inverse() * gt_cam_wrt_device_list[idx];
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert1_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert1_multiview_regi_ref_frame;
  FrameTransform vert1_ref_frame = FrameTransform::Identity();
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
    vert1_ref_frame(0, 3) = -rotcenter[0];
    vert1_ref_frame(1, 3) = -rotcenter[1];
    vert1_ref_frame(2, 3) = -rotcenter[2];

    vert1_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vert1_ref_frame, true);
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert2_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert2_multiview_regi_ref_frame;
  FrameTransform vert2_ref_frame = FrameTransform::Identity();
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
    vert2_ref_frame(0, 3) = -rotcenter[0];
    vert2_ref_frame(1, 3) = -rotcenter[1];
    vert2_ref_frame(2, 3) = -rotcenter[2];

    vert2_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vert2_ref_frame, true);
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert3_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert3_multiview_regi_ref_frame;
  FrameTransform vert3_ref_frame = FrameTransform::Identity();
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
    vert3_ref_frame(0, 3) = -rotcenter[0];
    vert3_ref_frame(1, 3) = -rotcenter[1];
    vert3_ref_frame(2, 3) = -rotcenter[2];

    vert3_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vert3_ref_frame, true);
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> vert4_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> vert4_multiview_regi_ref_frame;
  FrameTransform vert4_ref_frame = FrameTransform::Identity();
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
    vert4_ref_frame(0, 3) = -rotcenter[0];
    vert4_ref_frame(1, 3) = -rotcenter[1];
    vert4_ref_frame(2, 3) = -rotcenter[2];

    vert4_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(vert4_ref_frame, true);
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> sacrum_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> sacrum_multiview_regi_ref_frame;
  FrameTransform sacrum_ref_frame = FrameTransform::Identity();
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
    sacrum_ref_frame(0, 3) = -rotcenter[0];
    sacrum_ref_frame(1, 3) = -rotcenter[1];
    sacrum_ref_frame(2, 3) = -rotcenter[2];

    sacrum_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(sacrum_ref_frame, true);
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> spine_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> spine_multiview_regi_ref_frame;
  FrameTransform spine_ref_frame = FrameTransform::Identity();
  {
    vout << "setting up spine ref. frame..." << std::endl;
    // setup camera aligned reference frame, use metal spine volume center point as the origin

    itk::ContinuousIndex<double,3> center_idx;

    auto spine_fcsv_rotc = spine_3d_fcsv.find("vert3-cen");
    Pt3 spine_rotcenter;

    if (spine_fcsv_rotc != spine_3d_fcsv.end()){
      spine_rotcenter = spine_fcsv_rotc->second;
    }
    else{
      vout << "ERROR: NOT FOUND spine spine head center" << std::endl;
    }

    spine_singleview_regi_ref_frame = std::make_shared<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose>();
    spine_singleview_regi_ref_frame->vol_idx = 0;
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = spine_rotcenter[0];
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = spine_rotcenter[1];
    spine_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = spine_rotcenter[2];

    spine_ref_frame(0, 3) = -spine_rotcenter[0];
    spine_ref_frame(1, 3) = -spine_rotcenter[1];
    spine_ref_frame(2, 3) = -spine_rotcenter[2];

    spine_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(spine_ref_frame, true);
  }

  std::shared_ptr<MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose> device_singleview_regi_ref_frame;
  std::shared_ptr<MultiLevelMultiObjRegi::StaticRefFrame> device_multiview_regi_ref_frame;
  FrameTransform device_ref_frame = FrameTransform::Identity();
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

    device_ref_frame(0, 3) = -device_rotcenter[0];
    device_ref_frame(1, 3) = -device_rotcenter[1];
    device_ref_frame(2, 3) = -device_rotcenter[2];

    device_multiview_regi_ref_frame = MultiLevelMultiObjRegi::MakeStaticRefFrame(device_ref_frame, true);
  }

  FrameTransformList vert_refs = { vert1_ref_frame,
                                   vert2_ref_frame,
                                   vert3_ref_frame,
                                   vert4_ref_frame,
                                   sacrum_ref_frame};

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  CamModelList sim_cam;
  sim_cam = { default_cam };

  ProjDataF32List proj_spine_list;
  ProjDataF32List proj_device_list;
  proj_spine_list.reserve(num_views);
  proj_device_list.reserve(num_views);

  for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    ProjDataF32 proj_spine;
    ProjDataF32 proj_device;

    for (size_type device_flag = 0; device_flag < 2; device_flag++)
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_models(sim_cam);
      ray_caster->set_volumes({device_att, spine_att});
      // ray_caster->set_ray_step_size(0.5);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();

      if (device_flag == 1)
      {
        ray_caster->use_proj_store_replace_method();
        // TODO: change gt pose
        ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_device_list[view_idx]);
        ray_caster->compute(0);
        ray_caster->use_proj_store_accum_method();
      }
      // TODO: change gt pose
      ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_spine_list[view_idx]);
      ray_caster->compute(1);

      vout << "projecting view: " << view_idx << " device flag: " << device_flag << std::endl;

      if (device_flag == 1)
      {
        proj_device.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 10000);
        proj_device.cam = default_cam;
        proj_device_list.push_back( proj_device );
        if(kSAVE_REGI_DEBUG)
          WriteITKImageRemap8bpp(proj_device.img.GetPointer(), output_path + "/device" + std::to_string(view_idx) + ".png");
      }
      else
      {
        proj_spine.img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 10000);
        proj_spine.cam = default_cam;
        proj_spine_list.push_back( proj_spine );
        if(kSAVE_REGI_DEBUG)
          WriteITKImageRemap8bpp(proj_spine.img.GetPointer(), output_path + "/spine" + std::to_string(view_idx) + ".png");
      }
    }
  }

  std::vector<CameraModel> orig_cams;
  for ( size_type view_idx = 0; view_idx < num_views; ++view_idx)
  {
    orig_cams.push_back( default_cam );
  }

  // Using device regi as fiducial
  auto cams_devicefid = CreateCameraWorldUsingFiducial(orig_cams, gt_cam_wrt_device_list);
  FrameTransform fid_cam_wrt_device = gt_cam_wrt_device_list[0];

  for(size_type idx = 0;idx < 1000; ++idx)
  {
    const std::string exp_ID = fmt::format("{:04d}", idx);
    std::cout << "Running..." << exp_ID << std::endl;

    const std::string dst_h5_path = output_path + "/" + exp_ID + ".h5";
    H5::H5File dst_h5(dst_h5_path, H5F_ACC_TRUNC);

    const float rot_angX_rand = rot_angX_dist(rng_eng);
    const float rot_angY_rand = rot_angY_dist(rng_eng);
    const float rot_angZ_rand = rot_angZ_dist(rng_eng);

    vout << "Sampling left/right:" << rot_angY_rand << " back/forth:" << rot_angX_rand << " tilt:" << rot_angZ_rand << std::endl;
    RecomposeVertebrae recomp_vertb(meta_data_path, rot_angX_rand, rot_angY_rand, rot_angZ_rand);
    auto resample_vert = recomp_vertb.ResampleVertebrae(vout);
    const std::string resample_vert_vol_path = output_path + "/reample_vert" + exp_ID;
    if (kSAVE_REGI_DEBUG)
    {
      WriteITKImageToDisk(resample_vert.GetPointer(), resample_vert_vol_path + "_att.nii.gz");

      RecomposeVertebrae recomp_vertb_hu(meta_data_path, rot_angX_rand, rot_angY_rand, rot_angZ_rand, true);
      auto resample_vert_hu = recomp_vertb_hu.ResampleVertebrae(vout);
      WriteITKImageToDisk(resample_vert_hu.GetPointer(), resample_vert_vol_path + "_hu.nii.gz");
    }

    FrameTransformList ref_vert_xforms = { recomp_vertb.GetVert1Xform(),
                                           recomp_vertb.GetVert2Xform(),
                                           recomp_vertb.GetVert3Xform(),
                                           recomp_vertb.GetVert4Xform(),
                                           FrameTransform::Identity()};
    FrameTransformList delta_vert_xforms = {};
    for(size_type idx = 0; idx < 5; ++idx)
      delta_vert_xforms.push_back(vert_refs[idx].inverse() * ref_vert_xforms[idx] * vert_refs[idx]);

    FrameTransform init_cam_wrt_spine = spine_ref_frame.inverse() * Delta_rand_matrix(rot_mag_spine, trans_mag_spine) *  spine_ref_frame * gt_cam_wrt_spine_list[0];
    FrameTransform init_cam_wrt_device = device_ref_frame.inverse() * Delta_rand_matrix(rot_mag_device, trans_mag_device) * device_ref_frame * gt_cam_wrt_device_list[0];

    vout << "Setting up spine multi-view rigid registration..." << std::endl;
    MultiLevelMultiObjRegi regi_rigid;
    regi_rigid.set_debug_output_stream(vout, verbose);
    regi_rigid.set_save_debug_info(kSAVE_REGI_DEBUG);
    regi_rigid.vols = { resample_vert };
    regi_rigid.vol_names = { "spine" };

    regi_rigid.ref_frames = { spine_multiview_regi_ref_frame };

    regi_rigid.fixed_proj_data = proj_spine_list;

    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      regi_rigid.fixed_proj_data[view_idx].cam = cams_devicefid[view_idx];
    }

    regi_rigid.levels.resize(1);

    regi_rigid.init_cam_to_vols = { init_cam_wrt_spine * fid_cam_wrt_device.inverse() };

    auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();

    // Spine Single-view Registration
    {
      auto& lvl = regi_rigid.levels[0];

      lvl.ds_factor = 0.25;

      lvl.fixed_imgs_to_use.resize(num_views);
      std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

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

          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };
          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20) };

          cmaes_regi->set_pop_size(50);
          cmaes_regi->set_sigma({ 30 * kDEG2RAD, 30 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_coarse.regi = cmaes_regi;
        }
      }

      if (kSAVE_REGI_DEBUG )
      {
        // Create Debug Proj H5 File
        const std::string proj_data_h5_path = output_path + "/spine_singleview_proj_data" + exp_ID + ".h5";
        vout << "creating H5 proj data file for img" + exp_ID + "..." << std::endl;
        H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
        WriteProjDataH5(regi_rigid.fixed_proj_data, &h5);

        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
        debug_vol_path.vol_path = resample_vert_vol_path;

        regi_rigid.debug_info->vols = { debug_vol_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = proj_data_h5_path;
        // debug_proj_path.projs_used = { view_idx };

        regi_rigid.debug_info->fixed_projs = debug_proj_path;

        regi_rigid.debug_info->regi_names = { { "Singleview Spine" + exp_ID } };
      }
    }

    vout << std::endl << "Running Multiview spine registration ..." << std::endl;
    regi_rigid.run();
    regi_rigid.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    FrameTransform spine_rigid_regi_xform = regi_rigid.cur_cam_to_vols[0];
    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_spine" + exp_ID + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi_rigid.debug_info, dst_debug_path);
      WriteITKAffineTransform(output_path + "/spine_regi_xform" + exp_ID + ".h5", spine_rigid_regi_xform);
    }

    if (kSAVE_REGI_DEBUG)
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_model(default_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volume(spine_att);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->xform_cam_to_itk_phys(0) = spine_rigid_regi_xform * fid_cam_wrt_device;
      ray_caster->compute(0);

      WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/spine_reproj" + exp_ID + ".png");
    }

    vout << "Setting up vertb registration..." << std::endl;
    MultiLevelMultiObjRegi regi_vertb;
    regi_vertb.set_debug_output_stream(vout, verbose);
    regi_vertb.set_save_debug_info(kSAVE_REGI_DEBUG);
    regi_vertb.vols = { vert1_att, vert2_att, vert3_att, vert4_att, sacrum_att };
    regi_vertb.vol_names = { "vert1", "vert2", "vert3", "vert4", "sacrum" };

    regi_vertb.ref_frames = { vert1_multiview_regi_ref_frame, vert2_multiview_regi_ref_frame,
                              vert3_multiview_regi_ref_frame, vert4_multiview_regi_ref_frame,
                              sacrum_multiview_regi_ref_frame };

    regi_vertb.fixed_proj_data = proj_spine_list;

    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      regi_vertb.fixed_proj_data[view_idx].cam = cams_devicefid[view_idx];
    }

    regi_vertb.levels.resize(1);

    regi_vertb.init_cam_to_vols = {};
    for(size_type idx = 0; idx < 5; ++idx)
    {
      regi_vertb.init_cam_to_vols.push_back( delta_vert_xforms[idx] * spine_rigid_regi_xform );
    }

    // Spine Single-view Registration
    {
      auto& lvl = regi_vertb.levels[0];

      lvl.ds_factor = 0.25;

      lvl.fixed_imgs_to_use.resize(num_views);
      std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

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

          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(20 * kDEG2RAD, 20 * kDEG2RAD),
                                         std::make_shared<FoldNormDist>(20 * kDEG2RAD, 20 * kDEG2RAD),
                                         std::make_shared<FoldNormDist>(20 * kDEG2RAD, 20 * kDEG2RAD),
                                         std::make_shared<FoldNormDist>(20 * kDEG2RAD, 20 * kDEG2RAD),
                                         std::make_shared<FoldNormDist>(20 * kDEG2RAD, 20 * kDEG2RAD) };

          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(25, 25),
                                         std::make_shared<FoldNormDist>(40, 40),
                                         std::make_shared<FoldNormDist>(55, 55),
                                         std::make_shared<FoldNormDist>(70, 70),
                                         std::make_shared<FoldNormDist>(15, 15) };

          cmaes_regi->set_pop_size(50);
          cmaes_regi->set_sigma({ 5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 15, 15, 15,
                                  10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 20, 20, 20,
                                  15 * kDEG2RAD, 15 * kDEG2RAD, 15 * kDEG2RAD, 25, 25, 25,
                                  20 * kDEG2RAD, 20 * kDEG2RAD, 20 * kDEG2RAD, 30, 30, 30,
                                  5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 10, 10, 19 });

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
        WriteProjDataH5(regi_vertb.fixed_proj_data, &h5);

        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo vert1_path;
        vert1_path.vol_path = meta_data_path + "/vert1";

        DebugRegiResultsMultiLevel::VolPathInfo vert2_path;
        vert2_path.vol_path = meta_data_path + "/vert2";

        DebugRegiResultsMultiLevel::VolPathInfo vert3_path;
        vert3_path.vol_path = meta_data_path + "/vert3";

        DebugRegiResultsMultiLevel::VolPathInfo vert4_path;
        vert4_path.vol_path = meta_data_path + "/vert4";

        DebugRegiResultsMultiLevel::VolPathInfo sacrum_path;
        sacrum_path.vol_path = meta_data_path + "/sacrum";

        regi_vertb.debug_info->vols = { vert1_path, vert2_path, vert3_path, vert4_path, sacrum_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = proj_data_h5_path;
        // debug_proj_path.projs_used = { view_idx };

        regi_vertb.debug_info->fixed_projs = debug_proj_path;

        regi_vertb.debug_info->regi_names = { { "Singleview Vertb" + exp_ID } };
      }
    }

    vout << std::endl << "Multi-view vertb registration ..." << std::endl;
    regi_vertb.run();
    regi_vertb.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    FrameTransformList vertb_regi_xform_list = {};
    for(size_type idx = 0; idx < 5; ++idx)
    {
      vertb_regi_xform_list.push_back(regi_vertb.cur_cam_to_vols[idx]);
    }

    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_vertb" + exp_ID + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi_vertb.debug_info, dst_debug_path);
    }

    if (kSAVE_REGI_DEBUG)
    {

      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_models(sim_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volumes({vert1_att, vert2_att, vert3_att, vert4_att, sacrum_att});
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->use_proj_store_replace_method();
      for(size_type idx = 0; idx < 5; ++idx)
      {
        ray_caster->distribute_xform_among_cam_models(vertb_regi_xform_list[idx] * fid_cam_wrt_device);
        ray_caster->compute(idx);
        ray_caster->use_proj_store_accum_method();
      }

      WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/vert_reproj" + exp_ID + ".png");
    }

    vout << "Setting up device multi-view rigid registration..." << std::endl;
    MultiLevelMultiObjRegi regi_device;
    regi_device.set_debug_output_stream(vout, verbose);
    regi_device.set_save_debug_info(kSAVE_REGI_DEBUG);
    regi_device.vols = { device_att };
    regi_device.vol_names = { "device" };

    regi_device.ref_frames = { device_multiview_regi_ref_frame };

    regi_device.fixed_proj_data = proj_device_list;

    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      regi_device.fixed_proj_data[view_idx].cam = cams_devicefid[view_idx];
    }

    regi_device.levels.resize(1);

    regi_device.init_cam_to_vols = { init_cam_wrt_device * fid_cam_wrt_device.inverse() };
    // Device Multi-view Registration
    {
      auto& lvl = regi_device.levels[0];

      lvl.ds_factor = 0.25;

      lvl.fixed_imgs_to_use.resize(num_views);
      std::iota(lvl.fixed_imgs_to_use.begin(), lvl.fixed_imgs_to_use.end(), 0);

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

          pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };
          pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20) };

          cmaes_regi->set_pop_size(50);
          cmaes_regi->set_sigma({ 30 * kDEG2RAD, 30 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });

          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

          lvl_regi_coarse.regi = cmaes_regi;
        }
      }

      if (kSAVE_REGI_DEBUG )
      {
        // Create Debug Proj H5 File
        const std::string proj_data_h5_path = output_path + "/device_multiview_proj_data" + exp_ID + ".h5";
        vout << "creating H5 proj data file for img" + exp_ID + "..." << std::endl;
        H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);
        WriteProjDataH5(regi_device.fixed_proj_data, &h5);

        vout << "  setting regi debug info..." << std::endl;

        DebugRegiResultsMultiLevel::VolPathInfo debug_devicevol_path;
        debug_devicevol_path.vol_path = devicevol_path;

        if (use_seg)
        {
          debug_devicevol_path.label_vol_path = { deviceseg_path };
          debug_devicevol_path.labels_used    = { device_label };
        }

        regi_device.debug_info->vols = { debug_devicevol_path };

        DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
        debug_proj_path.path = proj_data_h5_path;
        // debug_proj_path.projs_used = { view_idx };

        regi_device.debug_info->fixed_projs = debug_proj_path;

        regi_device.debug_info->regi_names = { { "Multiview Device" + exp_ID } };
      }
    }

    vout << std::endl << "Running Multiview device registration ..." << std::endl;
    regi_device.run();
    regi_device.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    FrameTransform device_regi_xform = regi_device.cur_cam_to_vols[0];
    if (kSAVE_REGI_DEBUG)
    {
      vout << "writing debug info to disk..." << std::endl;
      const std::string dst_debug_path = output_path + "/debug_device" + exp_ID + ".h5";
      WriteMultiLevel2D3DRegiDebugToDisk(*regi_device.debug_info, dst_debug_path);
      WriteITKAffineTransform(output_path + "/device_regi_xform" + exp_ID + ".h5", device_regi_xform);
    }

    if (kSAVE_REGI_DEBUG)
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);
      ray_caster->set_camera_model(default_cam);
      ray_caster->use_proj_store_replace_method();
      ray_caster->set_volume(device_att);
      ray_caster->set_num_projs(1);
      ray_caster->allocate_resources();
      ray_caster->xform_cam_to_itk_phys(0) = device_regi_xform * fid_cam_wrt_device;
      ray_caster->compute(0);

      WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/device_reproj" + exp_ID + ".png");
    }

    // Log Regi Result Files
    {
      WriteAffineTransform4x4("gt-cam-wrt-spine", gt_cam_wrt_spine_list[0], &dst_h5);
      WriteAffineTransform4x4("gt-cam-wrt-device", gt_cam_wrt_device_list[0], &dst_h5);
      WriteAffineTransform4x4("init-cam-wrt-spine", init_cam_wrt_spine, &dst_h5);
      WriteAffineTransform4x4("init-cam-wrt-device", init_cam_wrt_device, &dst_h5);
      WriteAffineTransform4x4("fid-cam-wrt-device", fid_cam_wrt_device, &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-rigid-spine", spine_rigid_regi_xform, &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-device", device_regi_xform, &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-vert1", vertb_regi_xform_list[0], &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-vert2", vertb_regi_xform_list[1], &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-vert3", vertb_regi_xform_list[2], &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-vert4", vertb_regi_xform_list[3], &dst_h5);
      WriteAffineTransform4x4("regi-cam-wrt-sacrum", vertb_regi_xform_list[4], &dst_h5);
      WriteSingleScalarH5("vert-rotX", rot_angX_rand, &dst_h5);
      WriteSingleScalarH5("vert-rotY", rot_angY_rand, &dst_h5);
      WriteSingleScalarH5("vert-rotZ", rot_angZ_rand, &dst_h5);
      dst_h5.close();
    }
  }
  return 0;
}
