
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

// Manually annotated polaris fiducial marker center in each 2D X-ray image
std::vector< std::vector<float> > fid_center_annot{
  { 573.5, 1059.5, 799.5, 609.5, 1256.5, 892.5, 949.5, 1233.5 },
  { 513.5, 1067.5, 745.5, 593.5, 1193.5, 891.5, 894.5, 1243.5 },
  { 509.5, 1076.5, 736.5, 587.5, 1163.5, 890.5, 877.5, 1253.5 },
  { 611.5, 1079.5, 826.5, 580.5, 1217.5, 887.5, 954.5, 1257.5 },
  { 628.5, 1090.5, 828.5, 575.5, 1190.5, 892.5, 949.5, 1268.5 },
  { 900.5, 1082.5, 1066.5, 580.5, 1358.5, 885.5, 1161.5, 1255.5 },
  { 1064.5, 1068.5, 1198.5, 578.5, 1427.5, 876.5, 1271.5, 1237.5 },
  { 1132.5, 1069.5, 1224.5, 579.5, 1379.5, 875.5, 1271.5, 1232.5 },
  { 771.5, 894.5, 987.5, 477.5, 1421.5, 756.5, 1118.5, 1073.5 },
  { 306.5, 615.5, 505.5, 161.5, 978.5, 421.5, 688.5, 772.5 },
  { 516.5, 1085.5, 698.5, 611.5, 1161.5, 873.5, 903.5, 1238.5 },
  { 646.5, 652.5, 879.5, 188.5, 1305.5, 467.5, 1004.5, 813.5 },
  { 552.5, 1002.5, 804.5, 553.5, 1204.5, 810.5, 894.5, 1143.5 },
  { 453.5, 1009.5, 725.5, 540.5, 1085.5, 809.5, 774.5, 1154.5 },
  { 595.5, 1019.5, 865.5, 539.5, 1144.5, 810.5, 857.5, 1162.5 },
  { 572.5, 1029.5, 842.5, 525.5, 1045.5, 810.5, 776.5, 1177.5 },
  { 596.5, 1043.5, 859.5, 518.5, 986.5, 811.5, 744.5, 1193.5 },
  { 577.5, 848.5, 893.5, 351.5, 988.5, 650.5, 708.5, 1007.5 },
  { 655.5, 903.5, 976.5, 419.5, 1148.5, 704.5, 845.5, 1052.5 },
  { 656.5, 899.5, 974.5, 430.5, 1210.5, 707.5, 893.5, 1046.5 },
  { 685.5, 892.5, 993.5, 438.5, 1286.5, 708.5, 959.5, 1035.5 },
  { 755.5, 889.5, 1046.5, 451.5, 1395.5, 710.5, 1067.5, 1030.5 },
  { 801.5, 1033.5, 1061.5, 583.5, 1440.5, 840.5, 1130.5, 1174.5 },
  { 574.5, 1031.5, 817.5, 584.5, 1248.5, 842.5, 935.5, 1175.5 },
  { 562.5, 1019.5, 771.5, 594.5, 1228.5, 842.5, 931.5, 1163.5 },
  { 383.5, 1006.5, 540.5, 601.5, 994.5, 839.5, 734.5, 1145.5 },
  { 416.5, 1147.5, 570.5, 731.5, 1032.5, 981.5, 777.5, 1299.5 },
  { 519.5, 943.5, 626.5, 505.5, 1057.5, 667.5, 854.5, 1036.5 },
  { 387.5, 829.5, 448.5, 370.5, 917.5, 542.5, 740.5, 924.5 },
  { 530.5, 979.5, 643.5, 535.5, 1042.5, 700.5, 837.5, 1078.5} };

// 4 polaris marker center coordinates in marker origin frame
Pt3 polaris1_pt_3d = { 68.8633f,     68.7579f,      0.0337f }; // #1
Pt3 polaris2_pt_3d = { 74.7105f,     0.0000f,     -0.0252f };  // #2
Pt3 polaris3_pt_3d = { 0.0000f,      0.0000f,      0.0287f };  // #3
Pt3 polaris4_pt_3d = { 11.8127f,     62.1510f,     -0.0373f }; // #4

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Single view registration of the pelvis and femur. Femur registration is initialized by pelvis registration.");
  po.set_arg_usage("< Meta data path > < pelvis X-ray name txt file > < calibration X-ray name txt file > < pelvis registration xform > "
                   "< pelvis X-ray image DCM folder > < calibration X-ray image DCM folder > < handeyeX > < Tracker data folder >"
                   "< marker pnp xform folder > < output folder >");
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
  const std::string pelvis_regi_xform_path       = po.pos_args()[3];  // Pelvis registration transformation in 1st image view
  const std::string pelvis_dicom_path            = po.pos_args()[4];  // Pelvis Dicom X-ray image folder path
  const std::string calibration_dicom_path       = po.pos_args()[5];  // Dicom X-ray image folder path
  const std::string handeyeX_path                = po.pos_args()[6];  // Path to handeyeX matrix
  const std::string calibraion_tracker_path      = po.pos_args()[7];  // Path to calibration tracker data
  const std::string marker_pnp_path              = po.pos_args()[8];  // Path to pnp xform folder
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

  const std::string pel_xray_ID      = pelvis_xray_ID_list[0];
  const std::string pelvis_img_path  = pelvis_dicom_path + "/" + pel_xray_ID;

  const std::string cal_xray_01_ID   = calibration_xray_ID_list[0];
  const std::string cal_xray_02_ID   = calibration_xray_ID_list[1];
  const std::string cal_img_01_path  = calibration_dicom_path + "/" + cal_xray_01_ID;
  const std::string cal_img_02_path  = calibration_dicom_path + "/" + cal_xray_02_ID;

  // Find relative geometry between cal img 01 and 02
  auto handeyeX_xform = ReadITKAffineTransformFromFile(handeyeX_path);
  FrameTransformList cal_RB4_wrt_CarmFid_xform_list;
  FrameTransformList cal_pnp_xform_list;

  vout << "Reading calibration tracker data..." << std::endl;
  for(size_type cal_id = 0; cal_id < calibration_xray_ID_list.size(); ++cal_id)
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

    const std::string pnp_xform_path = marker_pnp_path + "/pnp_xform" + calibration_xray_ID_list[cal_id] + ".h5";
    FrameTransform pnp_xform = ReadITKAffineTransformFromFile(pnp_xform_path);
    cal_pnp_xform_list.push_back(pnp_xform);
  }

  FrameTransform rel_carm_xform = handeyeX_xform.inverse() * cal_RB4_wrt_CarmFid_xform_list[0].inverse() * cal_RB4_wrt_CarmFid_xform_list[1] * handeyeX_xform;
  const std::string rel_carm_xform_file = output_path + "/rel_carm_xform.h5";
  WriteITKAffineTransform(rel_carm_xform_file, rel_carm_xform);

  ProjDataF32 pel_img;
  ProjDataF32 cal_img01;
  ProjDataF32 cal_img02;
  std::vector<CIOSFusionDICOMInfo> pel_cios_metas(1), cal_cios_metas(1);
  {
    std::tie(pel_img.img, pel_cios_metas[0]) = ReadCIOSFusionDICOMFloat(pelvis_img_path);
    std::tie(cal_img01.img, cal_cios_metas[0]) = ReadCIOSFusionDICOMFloat(cal_img_01_path);
    std::tie(cal_img02.img, cal_cios_metas[0]) = ReadCIOSFusionDICOMFloat(cal_img_02_path);
  }

  // Artificually add two images together for the first calibration image
  vout << "Adding pelvis and calibration images ..." << std::endl;
  pel_img.img = ITKAddImages(pel_img.img.GetPointer(), cal_img01.img.GetPointer());
  WriteITKImageRemap8bpp(pel_img.img.GetPointer(), output_path + "/real" + pel_xray_ID + ".png");

  FrameTransform pelvis_regi_xform = ReadITKAffineTransformFromFile(pelvis_regi_xform_path);

  FrameTransform pelvis_carm2_xform = pelvis_regi_xform * rel_carm_xform;
  {
    auto ray_caster = LineIntRayCasterFromProgOpts(po);
    ray_caster->set_camera_model(default_cam);
    ray_caster->use_proj_store_replace_method();
    ray_caster->set_volume(vol_att);
    ray_caster->set_num_projs(1);
    ray_caster->allocate_resources();
    ray_caster->xform_cam_to_itk_phys(0) = pelvis_carm2_xform;
    ray_caster->compute(0);
    // pel_carm2_drr->SetSpacing(cal_img02.img->GetSpacing());

    // auto added_pel_cal_carm2 = ITKAddImages(pel_carm2_drr, cal_img02.img.GetPointer());
    WriteITKImageRemap8bpp(ray_caster->proj(0).GetPointer(), output_path + "/carm2_pelvis_drr.png");
    WriteITKImageRemap8bpp(cal_img02.img.GetPointer(), output_path + "/real_calibration_02.png");
  }

  //Reproject 3D marker landmark using 1st view pnp xform and 2nd view rel_carm_xform
  std::ofstream marker_reproj_file;
  marker_reproj_file.open(output_path + "/marker_reproj.txt", std::ios::trunc);

  for(size_type idx = 0; idx < calibration_xray_ID_list.size(); ++idx)
  {
    FrameTransform marker_xform;
    if(idx == 0)
      marker_xform = cal_pnp_xform_list[0];
    else
      marker_xform = cal_pnp_xform_list[0] * rel_carm_xform;

    Pt3 polaris1_reproj = default_cam.intrins * default_cam.extrins * marker_xform.inverse() * polaris1_pt_3d;
    Pt3 polaris2_reproj = default_cam.intrins * default_cam.extrins * marker_xform.inverse() * polaris2_pt_3d;
    Pt3 polaris3_reproj = default_cam.intrins * default_cam.extrins * marker_xform.inverse() * polaris3_pt_3d;
    Pt3 polaris4_reproj = default_cam.intrins * default_cam.extrins * marker_xform.inverse() * polaris4_pt_3d;

    // Difference of opencv convention image coordinate v.s. ITK
    marker_reproj_file << 1536 - polaris1_reproj(0)/polaris1_reproj(2) << ' ' << 1536 - polaris1_reproj(1)/polaris1_reproj(2)
                << ' ' << 1536 - polaris2_reproj(0)/polaris2_reproj(2) << ' ' << 1536 - polaris2_reproj(1)/polaris2_reproj(2)
                << ' ' << 1536 - polaris3_reproj(0)/polaris3_reproj(2) << ' ' << 1536 - polaris3_reproj(1)/polaris3_reproj(2)
                << ' ' << 1536 - polaris4_reproj(0)/polaris4_reproj(2) << ' ' << 1536 - polaris4_reproj(1)/polaris4_reproj(2)
                << '\n';
  }

  marker_reproj_file.close();

  // Transform Polaris points to pelvis CT frame
  {
    FrameTransform marker_xform = cal_pnp_xform_list[0];
    Pt3 polaris1_pelvis = pelvis_regi_xform * marker_xform.inverse() * polaris1_pt_3d;
    Pt3 polaris2_pelvis = pelvis_regi_xform * marker_xform.inverse() * polaris2_pt_3d;
    Pt3 polaris3_pelvis = pelvis_regi_xform * marker_xform.inverse() * polaris3_pt_3d;
    Pt3 polaris4_pelvis = pelvis_regi_xform * marker_xform.inverse() * polaris4_pt_3d;

    LandMap3 polaris_pts_map;
    polaris_pts_map.emplace("polaris1", polaris1_pelvis);
    polaris_pts_map.emplace("polaris2", polaris2_pelvis);
    polaris_pts_map.emplace("polaris3", polaris3_pelvis);
    polaris_pts_map.emplace("polaris4", polaris4_pelvis);

    ConvertRASToLPS(&polaris_pts_map);

    const std::string polaris_pelvis_fcsv_file = output_path + "/polaris_pts_pelvis.fcsv";
    WriteFCSVFileFromNamePtMap(polaris_pelvis_fcsv_file, polaris_pts_map);
  }

  return 0;
}
