// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>
#include <fmt/printf.h>

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregITKBasicImageUtils.h"
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
#include "xregIntensity2D3DRegiCMAES-JustinSnake.h"
#include "xregIntensity2D3DRegiBOBYQA.h"
#include "xregRegi2D3DPenaltyFnSE3Mag.h"
#include "xregRegi2D3DPenaltyFnSnakeLandReproj.h"
#include "xregFoldNormDist.h"
#include "xregNullDist.h"
#include "xregHipSegUtils.h"
#include "xregHDF5.h"
#include "xregHUToLinAtt.h"
#include "xregImageAddPoissonNoise.h"
#include "xregSampleUtils.h"
#include "xregSampleUniformUnitVecs.h"
#include "xregRotUtils.h"
#include "xregRigidUtils.h"

#include "bigssMath.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkVersor.h"
#include "itkChangeInformationImageFilter.h"

#include "xregBuildJustinSnakeModel.h"

#include "spline.h"
#include "bigssMath.h"

using namespace xreg;

constexpr int kEXIT_VAL_SUCCESS = 0;
constexpr int kEXIT_VAL_BAD_USE = 1;
constexpr bool kSAVE_REGI_DEBUG = true;
constexpr bool kSAVE_SNAKE_VOL_TO_DISK = false;

using size_type = std::size_t;
using VolScalar   = Float;
using LabelScalar = unsigned char;

using CamModel       = CameraModel;
using CamModelList   = std::vector<CameraModel>;

using Vol            = itk::Image<VolScalar, 3>;
using VolPtr         = Vol::Pointer;
using Seg            = itk::Image<LabelScalar, 3>;
using SegPtr         = Seg::Pointer;
using VolList        = std::vector<VolPtr>;

using SnakeVolumeType = itk::Image<Float, 3>;

using LandmarkList   = std::vector<int>;
using LandmarkList3D = std::vector<Pt3>;

using CamRefFrame    = xreg::MultiLevelMultiObjRegi::CamAlignRefFrameWithCurPose;

constexpr bool kDEBUG_SINGLE_VOL = false;
const size_type num_snake_vols = 28;
const std::string debug_data_path = "/home/cong/Research/Snake_Registration/Simulation/meta_data";

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

LandmarkList ROIcrop_landmark(LandmarkList snake_landmark)
{
  LandmarkList crop_landmark;

  size_type ROI_size =600;

  size_type min_x = snake_landmark[0] < snake_landmark[2] ? snake_landmark[0] : snake_landmark[2];
  size_type min_y = snake_landmark[1] < snake_landmark[3] ? snake_landmark[1] : snake_landmark[3];
  size_type max_x = snake_landmark[0] > snake_landmark[2] ? snake_landmark[0] : snake_landmark[2];
  size_type max_y = snake_landmark[1] > snake_landmark[3] ? snake_landmark[1] : snake_landmark[3];

  bool ld1_left = min_x == snake_landmark[0];
  bool ld1_up = min_y == snake_landmark[1];

  int edge_min_x = ld1_left ? ((min_x - 150) + (max_x + 50))/2 - ROI_size/2 : ((min_x - 50) + (max_x + 150))/2 - ROI_size/2;
  int edge_max_x = ld1_left ? ((min_x - 150) + (max_x + 50))/2 + ROI_size/2 : ((min_x - 50) + (max_x + 150))/2 + ROI_size/2;
  int edge_min_y = ld1_up ? ((min_y - 150) + (max_y + 50))/2 - ROI_size/2 : ((min_y - 50) + (max_y + 150))/2 - ROI_size/2;
  int edge_max_y = ld1_up ? ((min_y - 150) + (max_y + 50))/2 + ROI_size/2 : ((min_y - 50) + (max_y + 150))/2 + ROI_size/2;


  if(edge_min_x < 51)
  {
    int delta_x = 51 - edge_min_x + 1;
    min_x += delta_x;
    max_x += delta_x;
    edge_min_x += delta_x;
    edge_max_x += delta_x;
  }
  if(edge_max_x > 1485)
  {
    int delta_x = edge_max_x - 1485 + 1;
    min_x = min_x - delta_x;
    max_x = max_x - delta_x;
    edge_min_x -= delta_x;
    edge_max_x -= delta_x;
  }
  if(edge_min_y < 51)
  {
    int delta_y = 51 - edge_min_y + 1;
    min_y = min_y + delta_y;
    max_y = max_y + delta_y;
    edge_min_y += delta_y;
    edge_max_y += delta_y;
  }
  if(edge_max_y > 1485)
  {
    int delta_y = edge_max_y - 1485 + 1;
    min_y = min_y - delta_y;
    max_y = max_y - delta_y;
    edge_min_y -= delta_y;
    edge_max_y -= delta_y;
  }

  xregASSERT(edge_min_x > 50 && edge_min_y > 50 && edge_max_x < 1486 && edge_max_y < 1486);

  crop_landmark.clear();
  crop_landmark.push_back(edge_min_x);
  crop_landmark.push_back(edge_min_y);
  crop_landmark.push_back(edge_max_x);
  crop_landmark.push_back(edge_max_y);

  return crop_landmark;
}


FrameTransformList GetAllSegTransforms(FrameTransform snake_base_xform,
                                       std::vector<double> Y_ctr,
                                       FrameTransformList notch_ref_xform_list,
                                       Pt3List notch_rot_cen_list)
{
  FrameTransformList allseg_xforms = { snake_base_xform };

  tk::spline sp_interp;
  std::vector<double> X_ctr(5);

  for(size_type idx=0; idx<5; idx++){
    X_ctr[idx] = double(26.0*(idx+1)/6.0);
  }

  sp_interp.set_points(X_ctr, Y_ctr);

  Float accumX = 0.;
  Float accumY = 0.;
  Float transX = 0.;
  Float transY = 0.;
  Float rotZ = 0.;
  FrameTransform seg_rotation, snake_seg1_wrt_base;

  for (size_type vol_idx = 1; vol_idx < num_snake_vols; ++vol_idx)
  {
    FrameTransform cur_ref_xform = notch_ref_xform_list[vol_idx];

    rotZ += sp_interp(vol_idx);
    Float dist_rot_cen = vol_idx > 1 ? (notch_rot_cen_list[vol_idx-1] - notch_rot_cen_list[vol_idx-2]).norm() : 0.;//TODO: rot center list index
    accumX += dist_rot_cen * sin(rotZ * kDEG2RAD);
    accumY += dist_rot_cen * cos(rotZ * kDEG2RAD);
    Float rotcenX = notch_rot_cen_list[vol_idx-1][0] - notch_rot_cen_list[0][0];
    Float rotcenY = notch_rot_cen_list[vol_idx-1][1] - notch_rot_cen_list[0][1];
    Float accumX_wrt_rotcen = accumX - rotcenX;
    Float accumY_wrt_rotcen = accumY - rotcenY;
    transX = accumY_wrt_rotcen * sin(rotZ * kDEG2RAD) - accumX_wrt_rotcen * cos(rotZ * kDEG2RAD);
    transY = -(accumX_wrt_rotcen * sin(rotZ * kDEG2RAD) + accumY_wrt_rotcen * cos(rotZ * kDEG2RAD));
    transX = vol_idx % 2 == 0 ? transX + 0.03 : transX - 0.03;

    FrameTransform cur_vol_xform = cur_ref_xform.inverse() * EulerRotXYZTransXYZFrame(0, 0, rotZ * kDEG2RAD, transX, transY, 0) * cur_ref_xform * snake_base_xform;
    allseg_xforms.push_back(cur_vol_xform);
  }

  return allseg_xforms;
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("create xray geometries for each random fragment repositionings.");
  po.set_arg_usage(" < hdf5 file > < snake model path > < output path >");
  po.set_min_num_pos_args(3);

  po.add("save-debug-file", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "save-debug-file", "Save Debug files to disk output path.")
  << false;

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

  const std::string src_hdf5_path     = po.pos_args()[0];
  const std::string snake_model_path  = po.pos_args()[1];
  const std::string output_path       = po.pos_args()[2];

  const bool kSAVE_REGI_DEBUG         = po.get("save-debug-file");
  const size_type num_exp = 1;//5
  const size_type iter_mod = 10;
  const int radius = 5;

  const std::string snake_model_file = "/home/cong/Research/Snake_Registration/Simulation_JustinNewSnake/justin_output/built_model.nii.gz";
  size_type model_idx = 0;

  std::mt19937 rng_eng;
  SeedRNGEngWithRandDev(&rng_eng);

  vout << "opening input file for reading/writing..." << std::endl;
  H5::H5File h5(src_hdf5_path, H5F_ACC_RDWR);

  const std::string side_str = ReadStringH5("side", h5);

  const bool is_left = side_str == "left";

  auto vol_anatld_fcsv = ReadLandmarksMapH5Pt3(h5.openGroup("anat-landmarks"));

  // Read Snake 3D Landmarks
  auto snake_lands_fcsv = ReadFCSVFileNamePtMap(snake_model_path + "/JustinSnakeLandmarks.fcsv");
  ConvertRASToLPS(&snake_lands_fcsv);

  auto fem_fcsv = ReadLandmarksMapH5Pt3(h5.openGroup("fem-landmarks"));

  auto snake_ld1_fcsv_it = snake_lands_fcsv.find("ld1");
  auto snake_ld2_fcsv_it = snake_lands_fcsv.find("ld2");

  // Insert to 3D vol landmark map
  std::unordered_map<std::string, Pt3> vol_lands;
  LandmarkList3D lands_3d;

  if (snake_ld1_fcsv_it != snake_lands_fcsv.end())
  {
    std::pair<std::string, Pt3> ld1_3D("ld1", snake_ld1_fcsv_it->second);
    vol_lands.insert(ld1_3D);
    lands_3d.push_back(snake_ld1_fcsv_it->second);
  }
  else
  {
    xregThrow("ld1 not found!!!");
  }

  if (snake_ld2_fcsv_it != snake_lands_fcsv.end())
  {
    std::pair<std::string, Pt3> ld2_3D("ld2", snake_ld2_fcsv_it->second);
    vol_lands.insert(ld2_3D);
    lands_3d.push_back(snake_ld2_fcsv_it->second);
  }
  else
  {
    xregThrow("ld2 not found!!!");
  }

  FrameTransform app_to_ct = AnteriorPelvicPlaneFromLandmarksMap(vol_anatld_fcsv,
                            is_left ? kAPP_ORIGIN_LEFT_FH : kAPP_ORIGIN_RIGHT_FH);

  // shift the origin medial to get more of the pelvis in the FOV (x)
  // shift the origin down a little so we can fit more of the pelvis in the FOV (y)
  app_to_ct = app_to_ct * EulerRotXYZTransXYZFrame(0, 0, 0, (is_left ? -1 : 1) * 25, 35, 0);

  const FrameTransform ct_to_app = app_to_ct.inverse();

  auto ct_hu = ReadITKImageH5Float3D(h5.openGroup("vol"));

  auto ct_seg = ReadITKImageH5UChar3D(h5.openGroup("vol-seg"));

  h5.flush(H5F_SCOPE_GLOBAL);
  h5.close();

  auto ct_att = HUToLinAtt(ct_hu.GetPointer(), -130);

  const unsigned char pelvis_label = 1;
  const unsigned char femur_label = 3;

  vout << "extracting pelvis att. volume..." << std::endl;
  auto pelvis_vol = ApplyMaskToITKImage(ct_att.GetPointer(), ct_seg.GetPointer(), pelvis_label, Float(0), true);

  vout << "extracting femur att. volume..." << std::endl;
  auto femur_vol = ApplyMaskToITKImage(ct_att.GetPointer(), ct_seg.GetPointer(), femur_label, Float(0), true);

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  CamModelList cams;
  cams.reserve(1);
  cams.push_back(default_cam);

  //std::uniform_real_distribution<CoordScalar> intersect_pt_dist(0.65,0.8);

  // for sampling small perturbations of change in C-Arm view, so we do not have pure
  // orbital rotation
  std::uniform_real_distribution<Float> carm_small_rot_ang_dist(-2.0 * kDEG2RAD, 2.0 * kDEG2RAD);
  std::uniform_real_distribution<Float> carm_small_trans_mag_dist(-2.0, 2.0);

  std::shared_ptr<CamRefFrame> pelvis_singleview_regi_ref_frame;
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

    pelvis_singleview_regi_ref_frame = std::make_shared<CamRefFrame>();
    pelvis_singleview_regi_ref_frame->vol_idx = 0;
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = center_pt[0];
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = center_pt[1];
    pelvis_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = center_pt[2];
  }

  std::shared_ptr<CamRefFrame> femur_singleview_regi_ref_frame;
  {
    Pt3 femur_pt;

    auto check_label = [&ct_seg,&femur_pt,&vol_anatld_fcsv,femur_label] (const std::string& k)
    {
      bool found = false;

      auto vol_anatld_fcsv_it = vol_anatld_fcsv.find(k);

      if (vol_anatld_fcsv_it != vol_anatld_fcsv.end())
      {
        femur_pt = vol_anatld_fcsv_it->second;

        itk::Point<Float, 3> tmp_itk_pt;
        Seg::IndexType tmp_itk_idx;

        tmp_itk_pt[0] = femur_pt[0];
        tmp_itk_pt[1] = femur_pt[1];
        tmp_itk_pt[2] = femur_pt[2];

        ct_seg->TransformPhysicalPointToIndex(tmp_itk_pt, tmp_itk_idx);

        found = ct_seg->GetPixel(tmp_itk_idx) == femur_label;
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
      xregThrow("ERROR: could not find appropriate femur landmark!!");
    }

    femur_singleview_regi_ref_frame = std::make_shared<CamRefFrame>();
    femur_singleview_regi_ref_frame->vol_idx = 0;
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[0] = femur_pt[0];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[1] = femur_pt[1];
    femur_singleview_regi_ref_frame->center_of_rot_wrt_vol[2] = femur_pt[2];
  }

  /*
  H5::Group dst_projs_st = h5.openGroup("projs-softtissue");
  H5::Group dst_projs_dr = h5.openGroup("projs-snake");
  H5::Group control_pts_gt = h5.openGroup("control-pts-gt");
  H5::Group vol_xform_group = h5.openGroup("cam-wrt-vol-gt");
  H5::Group snake_xform_group = h5.openGroup("cam-wrt-snake-gt");
  */
  // ********** TMRB C-arm Projection Geometry ***************************************************************

  FrameTransform gt_cam_wrt_app = CreateAPViewOfAPP(default_cam, Float(0.8), true, is_left);

  FrameTransform pelvis_rot_matZ = EulerRotZFrame(-10 * kDEG2RAD);
  FrameTransform pelvis_rot_matY = EulerRotYFrame(-15 * kDEG2RAD);
  FrameTransform pelvis_rot_matX = EulerRotXFrame(-20 * kDEG2RAD);

  FrameTransform pelvis_rot_mat =  pelvis_rot_matZ * pelvis_rot_matY * pelvis_rot_matX;

  Pt3 pelvis_trans;
  pelvis_trans[0] = 0; pelvis_trans[1] = 10; pelvis_trans[2] = 30;

  FrameTransform delta_pelvis = FrameTransform::Identity();
  delta_pelvis.matrix().block(0,0,3,3) = pelvis_rot_mat.matrix().block(0,0,3,3);
  delta_pelvis.matrix().block(0,3,3,1) = pelvis_trans;

  FrameTransform delta_cam = FrameTransform::Identity();
  Pt3 cam_trans;
  cam_trans[0] = 0; cam_trans[1] = 30; cam_trans[2] = 10;
  delta_cam.matrix().block(0,3,3,1) = cam_trans;

  gt_cam_wrt_app = delta_pelvis * gt_cam_wrt_app * delta_cam;

  // ********** End TMRB C-arm Projection Geometry ***************************************************************

  // ********** Load Snake Segment Volume ************************************************************************
  std::vector<SnakeVolumeType::Pointer> snake_att_list;

  vout << "[Main] - Loading snake notch rotation centers from FCSV file..." << std::endl;
  const std::string notch_rot_cen_fcsv_path = snake_model_path + "/notch_rot_cen_Justin.fcsv";
  auto notch_rot_cen_fcsv = ReadFCSVFileNamePtMap(notch_rot_cen_fcsv_path);
  ConvertRASToLPS(&notch_rot_cen_fcsv);

  Pt3List notch_rot_cen_list;
  std::vector<std::shared_ptr<CamRefFrame>> snake_singleview_regi_ref_frame_list;
  FrameTransformList notch_ref_xform_list;
  std::vector<std::string> snake_vol_name_list;
  for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
  {
    auto vol_idx_name = fmt::format("{:03d}", vol_idx);
    vout << "   Loading snake vol: " << vol_idx << std::endl;

    snake_vol_name_list.push_back(vol_idx_name);

    auto snake_att = ReadITKImageFromDisk<SnakeVolumeType>(snake_model_path + "/" + vol_idx_name + "_att.nii.gz");
    snake_att_list.push_back(snake_att);

    Pt3 notch_rot_cen_pt;
    auto fcsv_finder = notch_rot_cen_fcsv.find(vol_idx_name);
    if(fcsv_finder != notch_rot_cen_fcsv.end())
    {
      notch_rot_cen_pt = fcsv_finder->second;
      notch_rot_cen_list.push_back(notch_rot_cen_pt);
    }
    else
    {
      std::cerr << "ERROR: NOT FOUND notch rotation center: " << vol_idx_name << std::endl;
    }

    std::shared_ptr<CamRefFrame> cur_regi_ref_frame = std::make_shared<CamRefFrame>();
    cur_regi_ref_frame->vol_idx = vol_idx;
    cur_regi_ref_frame->center_of_rot_wrt_vol[0] = notch_rot_cen_pt[0];
    cur_regi_ref_frame->center_of_rot_wrt_vol[1] = notch_rot_cen_pt[1];
    cur_regi_ref_frame->center_of_rot_wrt_vol[2] = notch_rot_cen_pt[2];

    snake_singleview_regi_ref_frame_list.push_back(cur_regi_ref_frame);

    FrameTransform cur_ref_frame = FrameTransform::Identity();
    {
      cur_ref_frame(0, 3) = -notch_rot_cen_pt[0];
      cur_ref_frame(1, 3) = -notch_rot_cen_pt[1];
      cur_ref_frame(2, 3) = -notch_rot_cen_pt[2];
    }

    notch_ref_xform_list.push_back(cur_ref_frame);
  }

  xregASSERT( snake_vol_name_list.size() == num_snake_vols );
  xregASSERT( snake_att_list.size() == num_snake_vols );
  xregASSERT( snake_singleview_regi_ref_frame_list.size() == num_snake_vols);
  xregASSERT( notch_ref_xform_list.size() == num_snake_vols);
  xregASSERT( notch_rot_cen_list.size() == num_snake_vols );

  FrameTransform snake_base_ref_frame = notch_ref_xform_list[0];

  // ********** Snake Geometry ******************************************************************
  FrameTransform xray_src_wrt_snakebase = FrameTransform::Identity();
  xray_src_wrt_snakebase.matrix().block(0,0,3,3) = EulerRotY(90 * kDEG2RAD) * EulerRotZ(90 * kDEG2RAD);
  xray_src_wrt_snakebase(0, 3) = -150;
  xray_src_wrt_snakebase(1, 3) = 50;
  xray_src_wrt_snakebase(2, 3) = 0;

  {
    std::vector<double> Y_ctr = {3.0, 3.0, 3.0, 3.0, 3.0};
    H5::H5File Yctr_h5file(snake_model_path + "/Y_ctr.h5", H5F_ACC_TRUNC);
    WriteVectorH5("Yctr", Y_ctr, &Yctr_h5file);
    Yctr_h5file.flush(H5F_SCOPE_GLOBAL);
    Yctr_h5file.close();
  }

  const std::string Yctr_file = snake_model_path + "/Y_ctr.h5";
  H5::H5File Yctr_h5file(Yctr_file, H5F_ACC_RDWR);
  auto Y_ctr = ReadVectorH5Double("Yctr", Yctr_h5file);
  Yctr_h5file.flush(H5F_SCOPE_GLOBAL);
  Yctr_h5file.close();

  // ********** Perform num_exp for each model file **************************************************************
  for (size_type exp_idx = 0; exp_idx < num_exp; ++exp_idx)
  {
    const std::string exp_ID = fmt::format("{:04d}", exp_idx);
    std::cout << "running exp " << exp_idx << std::endl;

    // ********** Randomized CT & snake Geometry *****************************************************************
    FrameTransform delta_app = FrameTransform::Identity();
    const FrameTransform gt_cam_wrt_fem = app_to_ct * delta_app * gt_cam_wrt_app;

    // ********** Randomized initial CT & Snake Geometry *********************************************************
    FrameTransform ct_delta_app = Delta_rand_matrix(3.0, 10.0);
    const FrameTransform init_cam_wrt_ct = app_to_ct * ct_delta_app * gt_cam_wrt_app;

    FrameTransform snake_delta_app = Delta_rand_matrix(10.0, 10.0);
    const FrameTransform init_cam_wrt_snakebase = snake_base_ref_frame.inverse() * snake_delta_app * snake_base_ref_frame * xray_src_wrt_snakebase;

    const FrameTransform gt_cam_wrt_snakemodel = xray_src_wrt_snakebase;

    // Initialize Snake Control Points
    std::vector<double> init_Y_ctr(5);
    std::array<Float, 5> std_ctr_pts = { 0.2, 0.4, 0.6, 0.6, 0.6};
    init_Y_ctr.clear();
    for(size_type ctr_idx=0; ctr_idx < 5; ++ctr_idx){
      init_Y_ctr.push_back(std::normal_distribution<double>(Y_ctr[ctr_idx], std_ctr_pts[ctr_idx])(rng_eng) - 1.0);
    }
    // Find Snake Reprojected landmarks
    FrameTransformList gt_allseg_xforms = GetAllSegTransforms(xray_src_wrt_snakebase, Y_ctr, notch_ref_xform_list, notch_rot_cen_list);
    const FrameTransform gt_cam_wrt_snakebase = gt_allseg_xforms[0];
    const FrameTransform gt_cam_wrt_snaketip  = gt_allseg_xforms[num_snake_vols-1];

    Pt3 ld1 = default_cam.phys_pt_to_ind_pt(Pt3(gt_allseg_xforms[0].inverse() *  lands_3d[0]));
    Pt3 ld2 = default_cam.phys_pt_to_ind_pt(Pt3(gt_allseg_xforms[num_snake_vols-1].inverse() * lands_3d[1]));

    int ld1_x = static_cast<int>(ld1[0] + 0.5);
    int ld1_y = static_cast<int>(ld1[1] + 0.5);
    int ld2_x = static_cast<int>(ld2[0] + 0.5);
    int ld2_y = static_cast<int>(ld2[1] + 0.5);

    LandmarkList snake_landmark = { ld1_x, ld1_y, ld2_x, ld2_y };

    auto crop_landmark = ROIcrop_landmark(snake_landmark);

    std::unordered_map<std::string, Pt3> corner_pts_fcsv;
    Pt3 corner1_pt = {Float(crop_landmark[0]), Float(crop_landmark[1]), 0.};
    Pt3 corner2_pt = {Float(crop_landmark[2]), Float(crop_landmark[3]), 0.};
    std::pair<std::string, Pt3> corner1("ld1", corner1_pt);
    std::pair<std::string, Pt3> corner2("ld2", corner2_pt);

    corner_pts_fcsv.insert(corner1);
    corner_pts_fcsv.insert(corner2);

    // Insert to 2D landmark map
    std::mt19937 rng_eng;
    SeedRNGEngWithRandDev(&rng_eng);

    // Add some noise for 2D landmark positions
    std::uniform_real_distribution<Float> land2d_noise_dist(-3, 3);

    // Insert to 2D landmark map
    std::unordered_map<std::string, Pt3> lds_crop;
    Pt3 ld1_pt_crop = {snake_landmark[0]-corner1_pt[0]+land2d_noise_dist(rng_eng), snake_landmark[1]-corner1_pt[1]+land2d_noise_dist(rng_eng), 0};
    Pt3 ld2_pt_crop = {snake_landmark[2]-corner1_pt[0]+land2d_noise_dist(rng_eng), snake_landmark[3]-corner1_pt[1]+land2d_noise_dist(rng_eng), 0};
    std::pair<std::string, Pt3> ld1_crop("ld1", ld1_pt_crop);
    std::pair<std::string, Pt3> ld2_crop("ld2", ld2_pt_crop);

    lds_crop.insert(ld1_crop);
    lds_crop.insert(ld2_crop);

    // ********** Generate Simulation Images *********************************************************************
    VolPtr orig_snakemetal_vol = ReadITKImageFromDisk<Vol>(snake_model_file);

    // HU value manipulation
    itk::ImageRegionIterator<Vol> it_metal(orig_snakemetal_vol, orig_snakemetal_vol->GetRequestedRegion());
    it_metal.GoToBegin();
    while( !it_metal.IsAtEnd()){
      if(it_metal.Get() == 1){
        it_metal.Set(8000);
      }
      else if(it_metal.Get() > 1){
        it_metal.Set(5000);
      }
      else if(it_metal.Get() == 0){
        it_metal.Set(-1000);
      }
      ++it_metal;
    }

    VolPtr snake_att = HUToLinAtt(orig_snakemetal_vol.GetPointer());

    ProjDataF32 proj_data_bone;
    ProjDataF32 proj_data_snake;

    for (size_type snake_flag = 0; snake_flag < 2; snake_flag++)
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);

      VolList proj_vols(2);
      proj_vols[0] = snake_att;
      proj_vols[1] = ct_att;

      ray_caster->set_camera_models(cams);
      ray_caster->set_num_projs(1);

      ray_caster->set_volumes(proj_vols);
      // ray_caster->set_ray_step_size(1.5);
      ray_caster->allocate_resources();
      if(snake_flag==1)
      {
        ray_caster->use_proj_store_replace_method();
        ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_snakemodel);
        ray_caster->compute(0);
        ray_caster->use_proj_store_accum_method();
      }
      ray_caster->distribute_xform_among_cam_models(gt_cam_wrt_fem);
      ray_caster->compute(1);

      // Caster::Pointer caster = Caster::New();
      if( snake_flag == 1)
      {
        proj_data_snake.img = CastITKImageIfNeeded<float>(SamplePoissonProjFromAttProj(ray_caster->proj(0).GetPointer(), 5000).GetPointer());
        // auto snake_save_img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 5000);
        // WriteITKImageRemap8bpp(proj_data_snake.img.GetPointer(), output_path+"/snake_regi_img_beforeproc" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + ".png");
        proj_data_snake.cam = default_cam;
      }
      else
      {
        proj_data_bone.img = CastITKImageIfNeeded<float>(SamplePoissonProjFromAttProj(ray_caster->proj(0).GetPointer(), 5000).GetPointer());
        proj_data_bone.cam = default_cam;
      }
    }


    vout << "preprocessing bone data..." << std::endl;
    ProjPreProc proj_bone_preproc;
    proj_bone_preproc.set_debug_output_stream(vout, verbose);
    proj_bone_preproc.input_projs = { proj_data_bone };
    proj_bone_preproc();
    proj_data_bone = proj_bone_preproc.output_projs[0];


    vout << "preprocessing snake data..." << std::endl;
    ProjPreProc proj_snake_preproc;
    proj_snake_preproc.set_debug_output_stream(vout, verbose);
    proj_snake_preproc.input_projs = { proj_data_snake };
    {
      auto& proj_lands = proj_snake_preproc.input_projs[0].landmarks;
      proj_lands.reserve(corner_pts_fcsv.size());

      for (const auto& fcsv_kv : corner_pts_fcsv)
      {
        proj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
      }
    }

    proj_snake_preproc();
    proj_data_snake = proj_snake_preproc.output_projs[0];

    if(kSAVE_REGI_DEBUG)
    {
      WriteITKImageRemap8bpp(proj_data_bone.img.GetPointer(), output_path+"/bone_regi_img" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + ".png");
      WriteITKImageRemap8bpp(proj_data_snake.img.GetPointer(), output_path+"/snake_regi_img" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + ".png");
    }

    ProjDataF32List proj_bone_list = { proj_data_bone };
    ProjDataF32List proj_snake_list = { proj_data_snake };

    // ********** Perfrom Pelvis & Femur Registration ******************************************************************
    MultiLevelMultiObjRegi regi_fem;

    regi_fem.set_debug_output_stream(vout, verbose);
    regi_fem.set_save_debug_info(kSAVE_REGI_DEBUG);

    regi_fem.vols = { pelvis_vol, femur_vol };
    regi_fem.vol_names = { "Pelvis", "Femur" };

    // CG: Reference frame inputs
    regi_fem.ref_frames = { pelvis_singleview_regi_ref_frame, femur_singleview_regi_ref_frame };
      // CG: readin fixed images
    regi_fem.fixed_proj_data = proj_bone_list;

    regi_fem.levels.resize(1);

    regi_fem.init_cam_to_vols = { init_cam_wrt_ct, init_cam_wrt_ct };

    femur_singleview_regi_ref_frame->cam_extrins = regi_fem.fixed_proj_data[0].cam.extrins;
    pelvis_singleview_regi_ref_frame->cam_extrins = regi_fem.fixed_proj_data[0].cam.extrins;

    auto se3_vars = std::make_shared<SE3OptVarsLieAlg>();
    auto so3_vars = std::make_shared<SO3OptVarsLieAlg>();

    {
      auto& lvl = regi_fem.levels[0];

      lvl.ds_factor = 0.125;

      lvl.fixed_imgs_to_use = { 0 };

      lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

      {
        vout << "setting up sim metric..." << std::endl;

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

      auto& lvl_regi_inten = lvl.regis[0];
      lvl_regi_inten.mov_vols = { 0 };
      lvl_regi_inten.static_vols = { };

      auto init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      init_guess->vol_idx = 0;
      lvl_regi_inten.init_mov_vol_poses = { init_guess };
      lvl_regi_inten.ref_frames = { 0 };
      {
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
        cmaes_regi->set_opt_vars(se3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();
        // CG: first view has larger search space

        pen_fn->rot_pdfs_per_obj   = { std::make_shared<FoldNormDist>(10 * kDEG2RAD, 10 * kDEG2RAD) };
        pen_fn->trans_pdfs_per_obj = { std::make_shared<FoldNormDist>(20, 20) };

        cmaes_regi->set_pop_size(50);
        cmaes_regi->set_sigma({ 30 * kDEG2RAD, 30 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 });

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi_inten.regi = cmaes_regi;
      }
    }

    if (kSAVE_REGI_DEBUG)
    {
      vout << "saving regi debug..." << std::endl;

      const std::string proj_data_bone_h5_path = output_path + "/pelvis_singleview_proj_data" + exp_ID + ".h5";
      vout << "creating H5 proj bone data file for exp" + exp_ID + "..." << std::endl;
      H5::H5File h5(proj_data_bone_h5_path, H5F_ACC_TRUNC);
      WriteProjDataH5(regi_fem.fixed_proj_data, &h5);

      DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
      debug_vol_path.vol_path = debug_data_path + "/18-2800_soft_1mm_crop.nii.gz";

      if (true)
      {
        debug_vol_path.label_vol_path = debug_data_path + "/18-2800_soft_1mm_crop_seg_touched_up.nii.gz";
        debug_vol_path.labels_used    = { pelvis_label, femur_label };
      }

      regi_fem.debug_info->vols = { debug_vol_path };

      DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
      debug_proj_path.path = proj_data_bone_h5_path;
      // debug_proj_path.projs_used = { view_idx };

      regi_fem.debug_info->fixed_projs = debug_proj_path;

      regi_fem.debug_info->regi_names = { { "Singleview Pelvis" + exp_ID } };
    }

    vout << std::endl << "Running pelvis registration ..." << std::endl;
    regi_fem.run();
    regi_fem.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    if (kSAVE_REGI_DEBUG)
    {
      vout << "saving pelvis regi debug..." << std::endl;
      WriteMultiLevel2D3DRegiDebugToDisk(*regi_fem.debug_info, output_path + "/pelvis_regi_debug" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + ".h5");
    }

    // ***************************** Femur Registration *****************************************
    auto& lvl = regi_fem.levels[0];

    lvl.ds_factor = 0.25;

    lvl.fixed_imgs_to_use = { 0 };

    lvl.ray_caster = LineIntRayCasterFromProgOpts(po);

    {
      vout << "setting up sim metric..." << std::endl;

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

    FrameTransform init_fem_xform = regi_fem.cur_cam_to_vols[0];
    regi_fem.init_cam_to_vols = { regi_fem.cur_cam_to_vols[0], init_fem_xform };

    {
      auto& lvl_regi = lvl.regis[0];
      lvl_regi.mov_vols = { 1 };
      lvl_regi.static_vols = { 0 };

      auto pel_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      pel_init_guess->vol_idx = 0;

      auto fem_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
      fem_init_guess->vol_idx = 1;

      lvl_regi.ref_frames = { 1 };

      lvl_regi.init_mov_vol_poses = { fem_init_guess };
      lvl_regi.static_vol_poses = { pel_init_guess };

      {
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAES>();
        cmaes_regi->set_opt_vars(so3_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        cmaes_regi->set_pop_size(50);
        cmaes_regi->set_sigma({ 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD });

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();
        pen_fn->rot_pdfs_per_obj = { std::make_shared<FoldNormDist>(45 * kDEG2RAD, 45 * kDEG2RAD) };
        pen_fn->trans_pdfs_per_obj = { std::make_shared<NullDist>(1) };

        cmaes_regi->set_penalty_fn(pen_fn);
        cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);

        lvl_regi.regi = cmaes_regi;
      }
    }

    if (kSAVE_REGI_DEBUG)
    {
      vout << "saving regi debug..." << std::endl;

      const std::string proj_data_bone_h5_path = output_path + "/femur_singleview_proj_data" + exp_ID + ".h5";
      vout << "creating H5 proj bone data file for exp" + exp_ID + "..." << std::endl;
      H5::H5File h5(proj_data_bone_h5_path, H5F_ACC_TRUNC);
      WriteProjDataH5(regi_fem.fixed_proj_data, &h5);

      DebugRegiResultsMultiLevel::VolPathInfo debug_vol_path;
      debug_vol_path.vol_path = debug_data_path + "/18-2800_soft_1mm_crop.nii.gz";

      if (true)
      {
        debug_vol_path.label_vol_path = debug_data_path + "/18-2800_soft_1mm_crop_seg_touched_up.nii.gz";
        debug_vol_path.labels_used    = { pelvis_label, femur_label };
      }

      regi_fem.debug_info->vols = { debug_vol_path };

      DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
      debug_proj_path.path = proj_data_bone_h5_path;
      // debug_proj_path.projs_used = { view_idx };

      regi_fem.debug_info->fixed_projs = debug_proj_path;

      regi_fem.debug_info->regi_names = { { "Singleview Femur" + exp_ID } };
    }

    vout << std::endl << " Femur registration ..." << std::endl;
    regi_fem.run();
    regi_fem.levels[0].regis[0].fns_to_call_right_before_regi_run.clear();

    if (kSAVE_REGI_DEBUG)
    {
      vout << "saving femur regi debug..." << std::endl;
      WriteMultiLevel2D3DRegiDebugToDisk(*regi_fem.debug_info, output_path + "/femur_regi_debug" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + ".h5");
    }

    FrameTransform est_cam_wrt_fem = regi_fem.cur_cam_to_vols[0];

    // ********** Perfrom Snake Registration ******************************************************************
    MultiLevelMultiObjRegi regi_snake;

    regi_snake.set_debug_output_stream(vout, verbose);
    regi_snake.set_save_debug_info(kSAVE_REGI_DEBUG);

    regi_snake.levels.resize(1);
    size_type view_idx = 0;
    auto snake_vars = std::make_shared<SnakeOptVars>();

    regi_snake.vols = snake_att_list;
    regi_snake.vol_names = snake_vol_name_list;

    // regi_snake.enable_debug_output();
    // regi_snake.save_debug_info = true;
    // regi_snake.debug_info.vol_path = debug_data_path + "/extendbase.mhd";
    // regi_snake.debug_info.regi_names = { { "Snake Regi" } };

    regi_snake.fixed_proj_data = proj_snake_list;

    for( auto cur_singleview_regi_ref_frame : snake_singleview_regi_ref_frame_list)
    {
      cur_singleview_regi_ref_frame->cam_extrins = regi_snake.fixed_proj_data[0].cam.extrins;
      regi_snake.ref_frames.push_back(cur_singleview_regi_ref_frame);
    }

    regi_snake.levels.resize(1);

    FrameTransformList init_allseg_xforms = GetAllSegTransforms(init_cam_wrt_snakebase, init_Y_ctr, notch_ref_xform_list, notch_rot_cen_list);

    regi_snake.init_cam_to_vols = { init_allseg_xforms };

    // Do a projection of all segments for debugging
    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);

      ray_caster->set_camera_models(cams);
      ray_caster->set_num_projs(1);

      ray_caster->set_volumes(regi_snake.vols);
      ray_caster->allocate_resources();
      for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
      {
        ray_caster->distribute_xform_among_cam_models( regi_snake.init_cam_to_vols[vol_idx] );
        ray_caster->compute(vol_idx);
        ray_caster->use_proj_store_accum_method();
      }

      auto proj_img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 5000);
      WriteITKImageRemap8bpp(proj_img.GetPointer(), output_path+"/snake_proj_img" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + ".png");
    }

    {
      auto ray_caster = LineIntRayCasterFromProgOpts(po);

      ray_caster->set_camera_models(cams);
      ray_caster->set_num_projs(1);

      ray_caster->set_volumes(regi_snake.vols);
      ray_caster->allocate_resources();
      for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
      {
        // ray_caster->use_proj_store_replace_method();
        auto encode_pt6 = ExpRigid4x4ToPt6(regi_snake.init_cam_to_vols[vol_idx].matrix());
        // vout << "   vol_idx:" << vol_idx << "   xform mat:\n" << regi_snake.init_cam_to_vols[vol_idx].matrix() << "\n   encode_pt6:\n" << encode_pt6 << std::endl;
        FrameTransform recon_xform;
        recon_xform.matrix() = ExpSE3(encode_pt6);
        ray_caster->distribute_xform_among_cam_models( recon_xform );
        ray_caster->compute(vol_idx);
        ray_caster->use_proj_store_accum_method();
      }

      auto proj_img = AddPoissonNoiseToImage(ray_caster->proj(0).GetPointer(), 5000);
      WriteITKImageRemap8bpp(proj_img.GetPointer(), output_path+"/snake_recon_img" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + ".png");
    }

    {
      auto& lvl = regi_snake.levels[0];
      lvl.ds_factor = 0.25;
      lvl.fixed_imgs_to_use = { 0 };

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
          patch_sm->set_patch_stride(5);
        }

        lvl.sim_metrics.push_back(sm);
      }

      lvl.regis.resize(1);

      auto& lvl_regi_coarse = lvl.regis[0];
      lvl_regi_coarse.mov_vols = { };
      for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx){
        lvl_regi_coarse.mov_vols.push_back( vol_idx );
        lvl_regi_coarse.ref_frames.push_back( vol_idx );
      }

      lvl_regi_coarse.static_vols = { };

      for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx){
        auto cur_init_guess = std::make_shared<MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst>();
        cur_init_guess->vol_idx = vol_idx;
        lvl_regi_coarse.init_mov_vol_poses.push_back( cur_init_guess );
      }

      {
        auto cmaes_regi = std::make_shared<Intensity2D3DRegiCMAESjustinsnake>();
        cmaes_regi->set_opt_vars(snake_vars);
        cmaes_regi->set_opt_x_tol(0.01);
        cmaes_regi->set_opt_obj_fn_tol(0.01);

        auto pen_fn = std::make_shared<Regi2D3DPenaltyFnSE3Mag>();
        pen_fn->rot_pdfs_per_obj = { std::make_shared<FoldNormDist>(45 * kDEG2RAD, 45 * kDEG2RAD) };
        pen_fn->trans_pdfs_per_obj = { std::make_shared<NullDist>(1) };

        cmaes_regi->set_pop_size(50);
        cmaes_regi->set_sigma({ 2 * kDEG2RAD, 2 * kDEG2RAD, 2 * kDEG2RAD, 5, 5, 5, 1.5, 1.5, 1.5, 1.5, 1.5});
        cmaes_regi->set_ctr_mean({Float(init_Y_ctr[0]), Float(init_Y_ctr[1]), Float(init_Y_ctr[2]), Float(init_Y_ctr[3]), Float(init_Y_ctr[4])});
        cmaes_regi->set_Yctr_save_path(output_path);
        cmaes_regi->set_notch_rot_cen_list(notch_rot_cen_list);
        cmaes_regi->set_notch_ref_xform_list(notch_ref_xform_list);

        lvl_regi_coarse.fns_to_call_right_before_regi_run.clear();

        if(true)
        {
          vout << "  using landmark re-proj penalty" << std::endl;

          auto land_pen_fn = std::make_shared<Regi2D3DPenaltyFnSnakeLandReproj>();

          land_pen_fn->std_dev = 100 * lvl.ds_factor;

          {
            std::unordered_map<std::string, Pt2> ds_lands_2d;

            Pt2 tmp_pt2;

            for (auto& lkv : lds_crop)
            {
              tmp_pt2[0] = lkv.second[0] * lvl.ds_factor;
              tmp_pt2[1] = lkv.second[1] * lvl.ds_factor;

              vout << "     " << lkv.first << ": " << tmp_pt2[0] << ", " << tmp_pt2[1] << std::endl;

              ds_lands_2d.emplace(lkv.first, tmp_pt2);
            }

            land_pen_fn->set_lands(vol_lands, ds_lands_2d);
          }

          cmaes_regi->set_penalty_fn(land_pen_fn);
        }

        /*
        if (!pen_fn->pen_fns.empty())
        {
          cmaes_regi->set_penalty_fn(pen_fn);
          cmaes_regi->set_img_sim_penalty_coefs(0.9, 1.0);
        }
        */
        lvl_regi_coarse.regi = cmaes_regi;
      }
    }

    if (kSAVE_REGI_DEBUG)
    {
      const std::string proj_data_snake_h5_path = output_path + "/snake_singleview_proj_data" + exp_ID + ".h5";
      vout << "creating H5 proj bone data file for exp" + exp_ID + "..." << std::endl;
      H5::H5File h5(proj_data_snake_h5_path, H5F_ACC_TRUNC);
      WriteProjDataH5(regi_snake.fixed_proj_data, &h5);

      for(size_type vol_idx = 0; vol_idx < num_snake_vols; ++vol_idx)
      {
        DebugRegiResultsMultiLevel::VolPathInfo cur_debug_path;
        cur_debug_path.vol_path = snake_model_path + "/" + snake_vol_name_list[vol_idx];
        regi_snake.debug_info->vols.push_back(cur_debug_path);
      }

      DebugRegiResultsMultiLevel::ProjDataPathInfo debug_proj_path;
      debug_proj_path.path = proj_data_snake_h5_path;

      regi_snake.debug_info->fixed_projs = debug_proj_path;

      regi_snake.debug_info->regi_names = { { "Singleview Snake" + exp_ID } };
    }

    regi_snake.run();

    if (kSAVE_REGI_DEBUG)
    {
      vout << "saving snake regi debug..." << std::endl;
      WriteMultiLevel2D3DRegiDebugToDisk(*regi_snake.debug_info, output_path + "/snake_regi_debug.h5");//" + fmt::sprintf("%03lu", model_idx) + "_exp" + fmt::sprintf("%03lu", exp_idx) + "
    }

    FrameTransform est_cam_wrt_snakebase = regi_snake.cur_cam_to_vols[0];
    FrameTransform est_cam_wrt_snaketip  = regi_snake.cur_cam_to_vols[num_snake_vols-1];

      // ********** Calculate Registration Accuracy ******************************************************************
    {
      vout << "calculating registration accuracy..." << std::endl;
      auto fem_fcsv_lsc = fem_fcsv.find("LSC");
      auto fem_fcsv_lep = fem_fcsv.find("LEP");

      Pt3 femur_lsc_pt;
      Pt3 femur_lep_pt;
      // Read in femur fiducial points
      if (fem_fcsv_lsc != fem_fcsv.end()){
        femur_lsc_pt = fem_fcsv_lsc->second;
      }
      else{
        vout << "ERROR: NOT FOUND LSC" << std::endl;
      }

      if (fem_fcsv_lep != fem_fcsv.end()){
        femur_lep_pt = fem_fcsv_lep->second;
      }
      else{
        vout << "ERROR: NOT FOUND LEP" << std::endl;
      }

      Pt3 femur_lep_cam_gt   = gt_cam_wrt_fem.inverse() * femur_lep_pt;
      Pt3 femur_lep_cam_pre  = est_cam_wrt_fem.inverse() * femur_lep_pt;
      Pt3 femur_lep_diff     = femur_lep_cam_gt - femur_lep_cam_pre;

      vout << "femur lep diff:" << femur_lep_diff.norm() << std::endl;

      Pt3 snaketip_cam_gt   = gt_cam_wrt_snaketip.inverse() * lands_3d[1];
      Pt3 snaketip_cam_pre  = est_cam_wrt_snaketip.inverse() * lands_3d[1];
      Pt3 snaketip_cam_diff = snaketip_cam_gt - snaketip_cam_pre;

      vout << "snaketip diff:" << snaketip_cam_diff.norm() << std::endl;

      Pt3 snaketip_wrt_lep_gt    = snaketip_cam_gt - femur_lep_cam_gt;
      Pt3 snaketip_wrt_lep_pre   = snaketip_cam_pre - femur_lep_cam_pre;
      Pt3 snaketip_wrt_lep_diff  = snaketip_wrt_lep_gt - snaketip_wrt_lep_pre;

      vout << "snaketip wrt lep diff:" << snaketip_wrt_lep_diff.norm() << std::endl;

      const FrameTransform pre_snakeref_wrt_cam   = snake_base_ref_frame * est_cam_wrt_snakebase;
      const FrameTransform gt_snakeref_wrt_cam    = snake_base_ref_frame * gt_cam_wrt_snakebase;
      const FrameTransform err_snakeref_xform =  pre_snakeref_wrt_cam * gt_snakeref_wrt_cam.inverse();

      /*
      H5::H5File h5_mean(output_path + "/mean.h5", H5F_ACC_RDWR);
      auto est_Y_ctr = ReadVectorH5Double("mean", h5_mean);
      h5_mean.flush(H5F_SCOPE_GLOBAL);
      h5_mean.close();

      tk::spline sp_interp_gt, sp_interp_est;
      std::vector<double> X_ctr(5);

      for(size_type idx=0; idx<5; idx++){
        X_ctr[idx] = double(26.0*(idx+1)/6.0);
      }

      sp_interp_gt.set_points(X_ctr, Y_ctr);
      sp_interp_est.set_points(X_ctr, est_Y_ctr);
      */
    }

    vout << "exiting..." << std::endl;
  }

  return kEXIT_VAL_SUCCESS;
}
