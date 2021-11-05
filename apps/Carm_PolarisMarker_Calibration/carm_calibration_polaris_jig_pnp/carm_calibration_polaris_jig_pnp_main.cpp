
// STD
#include <iostream>
#include <vector>

#include <fmt/format.h>
#include <opencv2/opencv.hpp>

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

  po.set_help("Device pose calculation using PnP solution");
  po.set_arg_usage("<exp ID list file path> <result folder path>");
  po.set_min_num_pos_args(1);

  po.add("debug-pnp", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "debug-pnp", "Debug PnP reprojection error")
  << false;
  po.add("write-pnp-xform", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "write-pnp-xform", "Write pnp xform to file")
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


  const std::string exp_list_path   = po.pos_args()[0];  // Exp ID list path
  const std::string result_path     = po.pos_args()[1];  // Output path

  const bool write_pnp_xform = po.get("write-pnp-xform");
  const bool debug_pnp = po.get("debug-pnp");

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

  // Create 3D polaris markers
  Pt3 polaris1_pt_3d = { 68.8633f,     68.7579f,      0.0337f }; // #1
  Pt3 polaris2_pt_3d = { 74.7105f,     0.0000f,     -0.0252f };  // #2
  Pt3 polaris3_pt_3d = { 0.0000f,      0.0000f,      0.0287f };  // #3
  Pt3 polaris4_pt_3d = { 11.8127f,     62.1510f,     -0.0373f }; // #4

  std::pair<std::string, Pt3> polaris_ld1_3d( "polaris1", polaris1_pt_3d );
  std::pair<std::string, Pt3> polaris_ld2_3d( "polaris2", polaris2_pt_3d );
  std::pair<std::string, Pt3> polaris_ld3_3d( "polaris3", polaris3_pt_3d );
  std::pair<std::string, Pt3> polaris_ld4_3d( "polaris4", polaris4_pt_3d );

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  for(int idx=0; idx < lineNumber; ++idx)
  {
    const std::string exp_ID  = exp_ID_list[idx];
    std::cout << "Running..." << exp_ID << std::endl;

    Pt2 polaris1_pt_2d = { 1536 - fid_center_annot[idx][0], 1536 - fid_center_annot[idx][1] };
    Pt2 polaris2_pt_2d = { 1536 - fid_center_annot[idx][2], 1536 - fid_center_annot[idx][3] };
    Pt2 polaris3_pt_2d = { 1536 - fid_center_annot[idx][4], 1536 - fid_center_annot[idx][5] };
    Pt2 polaris4_pt_2d = { 1536 - fid_center_annot[idx][6], 1536 - fid_center_annot[idx][7] };

    std::pair<std::string, Pt2> polaris_ld1_2d( "polaris1", polaris1_pt_2d );
    std::pair<std::string, Pt2> polaris_ld2_2d( "polaris2", polaris2_pt_2d );
    std::pair<std::string, Pt2> polaris_ld3_2d( "polaris3", polaris3_pt_2d );
    std::pair<std::string, Pt2> polaris_ld4_2d( "polaris4", polaris4_pt_2d );

    LandMap2 lands_2d;
    LandMap3 lands_3d;

    lands_2d.insert( polaris_ld1_2d );
    lands_2d.insert( polaris_ld2_2d );
    lands_2d.insert( polaris_ld3_2d );
    lands_2d.insert( polaris_ld4_2d );

    lands_3d.insert( polaris_ld1_3d );
    lands_3d.insert( polaris_ld2_3d );
    lands_3d.insert( polaris_ld3_3d );
    lands_3d.insert( polaris_ld4_3d );

    /* Solve it using OpenCV PnP */

    std::vector<cv::Point2d> image_points;
    // Difference of opencv convention image coordinate v.s. ITK
    image_points.push_back( cv::Point2d(1536 - fid_center_annot[idx][0], fid_center_annot[idx][1]) );    // polaris1
    image_points.push_back( cv::Point2d(1536 - fid_center_annot[idx][2], fid_center_annot[idx][3]) );    // polaris2
    image_points.push_back( cv::Point2d(1536 - fid_center_annot[idx][4], fid_center_annot[idx][5]) );    // polaris3
    image_points.push_back( cv::Point2d(1536 - fid_center_annot[idx][6], fid_center_annot[idx][7]) );    // polaris4

    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d( 68.8633f,     68.7579f,      0.0337f ));           // polaris1
    model_points.push_back(cv::Point3d( 74.7105f,     0.0000f,     -0.0252f ));           // polaris2
    model_points.push_back(cv::Point3d( 0.0000f,      0.0000f,      0.0287f ));           // polaris3
    model_points.push_back(cv::Point3d( 11.8127f,     62.1510f,     -0.0373f ));           // polaris4

    // Camera internals
    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << 5257.732,   0,   767.5,  0, 5257.732,  767.5, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;

    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

    // Check image points using opencv reprojection
    if(false)
    {
      std::vector<cv::Point2d> check_image_points;
      cv::projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs, check_image_points);
      std::cout << check_image_points << std::endl;
    }

    cv::Mat rotation_front;
    cv::Rodrigues(rotation_vector, rotation_front);

    FrameTransform cor_trans = FrameTransform::Identity();
    cor_trans(0,0) = 1; cor_trans(0,1) = 0; cor_trans(0,2) = 0;
    cor_trans(1,0) = 0; cor_trans(1,1) = -1; cor_trans(1,2) = 0;
    cor_trans(2,0) = 0; cor_trans(2,1) = 0; cor_trans(2,2) = -1;

    FrameTransform init_lands_vol_to_src = FrameTransform::Identity();
    for(size_type i=0; i<3; ++i)
    {
      for(size_type j=0; j<3; ++j)
      {
        init_lands_vol_to_src(i,j) = rotation_front.at<double>(i,j);
      }
      init_lands_vol_to_src(i,3) = translation_vector.at<double>(i);
    }

    FrameTransform Opencv_lands_cam_to_vol = init_lands_vol_to_src.inverse() * cor_trans.inverse() * default_cam.extrins;

    FrameTransform lands_cam_to_vol = PnPReprojCMAES(default_cam, lands_3d, lands_2d, Opencv_lands_cam_to_vol);

    if(write_pnp_xform)
      WriteITKAffineTransform(result_path + "/pnp_xform" + exp_ID + ".h5", lands_cam_to_vol);

    if(debug_pnp)
    {
      std::ofstream pnp_reproj_file;
      if(idx == 0)
        pnp_reproj_file.open(result_path + "/pnpreproj.txt", std::ios::trunc);
      else
        pnp_reproj_file.open(result_path + "/pnpreproj.txt", std::ios::app);

      Pt3 polaris1_reproj = default_cam.intrins * default_cam.extrins * lands_cam_to_vol.inverse() * polaris1_pt_3d;
      Pt3 polaris2_reproj = default_cam.intrins * default_cam.extrins * lands_cam_to_vol.inverse() * polaris2_pt_3d;
      Pt3 polaris3_reproj = default_cam.intrins * default_cam.extrins * lands_cam_to_vol.inverse() * polaris3_pt_3d;
      Pt3 polaris4_reproj = default_cam.intrins * default_cam.extrins * lands_cam_to_vol.inverse() * polaris4_pt_3d;

      pnp_reproj_file << exp_ID << ' ' << polaris1_reproj(0)/polaris1_reproj(2) << ' ' << polaris1_reproj(1)/polaris1_reproj(2)
                                << ' ' << polaris2_reproj(0)/polaris2_reproj(2) << ' ' << polaris2_reproj(1)/polaris2_reproj(2)
                                << ' ' << polaris3_reproj(0)/polaris3_reproj(2) << ' ' << polaris3_reproj(1)/polaris3_reproj(2)
                                << ' ' << polaris4_reproj(0)/polaris4_reproj(2) << ' ' << polaris4_reproj(1)/polaris4_reproj(2)
                                << '\n';
    }
  }

  return 0;
}
