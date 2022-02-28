
// STD
#include <iostream>
#include <vector>
#include <string>

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

const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

Pt3 ReadKwireAnnotationPtFromTxt(const std::string kwire_annot_txt_path)
{
  std::vector<int> annot_pt;
  {
    std::ifstream KwireAnnotFile(kwire_annot_txt_path);
    // Make sure the file is open
    if(!KwireAnnotFile.is_open()) throw std::runtime_error("Could not Kwire Annotation file");

    std::string line, csvItem;
    size_type lineNumber = 0;
    while(std::getline(KwireAnnotFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          annot_pt.push_back(stoi(csvItem));
      }
      lineNumber++;
    }
    // It's 2D annotation
    xregASSERT(lineNumber == 2);
  }

  Pt3 Kwire_pt_2d = {1536 - annot_pt[0], 1536 - annot_pt[1], 0};

  return Kwire_pt_2d;
}

Pt3 Brute_Force_Search_Pt(FrameTransformList kwire_pt_wrt_cam_list, std::vector<Pt3> Kwire_pt_2d_list, size_type num_views,
                          float range, float step)
{
  float cur_loss = 0.0;
  float tot_loss = 100000;
  Pt3 offset(3);

  // Perform Searching
  for(float dx = -range; dx < range; dx += step)
  {
    for(float dy = -range; dy < range; dy += step)
    {
      for(float dz = -range; dz < range; dz += step)
      {
        cur_loss = 0.0;
        // Do reprojection and calculate loss
        for(int idx = 0; idx < num_views; ++idx)
        {
          Pt3 search_kwire_pt_3d = {dx, dy, dz};
          Pt3 kwire_pt_reproj_3d = default_cam.intrins * default_cam.extrins * kwire_pt_wrt_cam_list[idx].inverse() * search_kwire_pt_3d;
          Pt3 kwire_pt_reproj = {0, 0, 0};
          kwire_pt_reproj[0] = kwire_pt_reproj_3d[0] / kwire_pt_reproj_3d[2];
          kwire_pt_reproj[1] = kwire_pt_reproj_3d[1] / kwire_pt_reproj_3d[2];
          //std::cout << idx << ' ' << kwire_pt_reproj[0] << ' ' << kwire_pt_reproj[1] << std::endl;
          cur_loss += (kwire_pt_reproj - Kwire_pt_2d_list[idx]).norm();
        }
        if(cur_loss < tot_loss)
        {
          tot_loss = cur_loss;
          offset[0] = dx;
          offset[1] = dy;
          offset[2] = dz;
        }
      }
    }
  }

  std::cout << "total loss:" << tot_loss << " dx:" << offset[0] << " dy:" << offset[1] << " dz:" << offset[2] << std::endl;

  return offset;
}

LandMap3 ReprojSearchePts(FrameTransform kwire_pt_wrt_cam, Pt3 offset)
{
  LandMap3 reproj_bbs_fcsv;
  auto reproj_bb = default_cam.phys_pt_to_ind_pt(kwire_pt_wrt_cam.inverse() * offset);
  reproj_bb[0] = 0.194 * (reproj_bb[0] - 1536);
  reproj_bb[1] = 0.194 * (reproj_bb[1] - 1536);
  reproj_bb[2] = 0;
  std::pair<std::string, Pt3> ld2_3D("Reproj", reproj_bb);
  reproj_bbs_fcsv.insert(ld2_3D);

  return reproj_bbs_fcsv;
}

FrameTransform SetRefFrame(Pt3 ref_pt)
{
  FrameTransform ref_frame = FrameTransform::Identity();

  ref_frame.matrix()(0,3) = -ref_pt[0];
  ref_frame.matrix()(1,3) = -ref_pt[1];
  ref_frame.matrix()(2,3) = -ref_pt[2];

  return ref_frame;
}

int main(int argc, char* argv[])
{
  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Brute Force search for fiducial point reconstruction. Reproject estimated fiducial point on each 2D image plane and minimize the 2D distance loss.");
  po.set_arg_usage("< Fiducial (such as pelvis) xform folder > < Kwire annotation text file folder >"
                   "< Img ID text file > < Initial Kwire Point FCSV >"
                   "< Triangulated point fcsv file > < output folder>");
  po.set_min_num_pos_args(6);

  po.add("initname", 'n', ProgOpts::kSTORE_STRING, "kwire-pt-name",
         "Kwire point initial name. Default is init")
    << "init";

  po.add("poseprefix", 'p', ProgOpts::kSTORE_STRING, "carm-pose-xform-prefix",
          "Carm pose estimation object xform name prefix. Default is pelvis_sv_regi_xform")
    << "pelvis_sv_regi_xform";

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

  const std::string fid_xform_folder_path        = po.pos_args()[0]; // Fiducial such as pelvis pose xform folder path
  const std::string kwire_annot_folder_path      = po.pos_args()[1]; // Folder containing kwire annotation txt files
  const std::string img_ID_list_path             = po.pos_args()[2]; // image ID list path
  const std::string kwire_pt_fcsv_path           = po.pos_args()[3]; // Initial tip position landmark
  const std::string triangulated_fcsv            = po.pos_args()[4]; // Output triangulated fcsv file
  const std::string output_path                  = po.pos_args()[5]; // Output folder

  const std::string kwire_pt_name = po.get("kwire-pt-name");
  const std::string carm_pose_xform_prefix = po.get("carm-pose-xform-prefix");

  std::cout << "reading init device kwire_pt landmark from FCSV file..." << std::endl;
  auto kwire_pt_3dfcsv = ReadFCSVFileNamePtMap(kwire_pt_fcsv_path);
  ConvertRASToLPS(&kwire_pt_3dfcsv);

  Pt3 kwire_pt_3d;
  {
    auto kwire_pt_ref_fcsv = kwire_pt_3dfcsv.find(kwire_pt_name);

    if (kwire_pt_ref_fcsv != kwire_pt_3dfcsv.end()){
      kwire_pt_3d = kwire_pt_ref_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND KWIRE REF PT" << std::endl;
    }
  }

  std::vector<std::string> img_ID_list;
  int lineNumber = 0;
  /* Read exp ID list from file */
  {
    std::ifstream expIDFile(img_ID_list_path);
    // Make sure the file is open
    if(!expIDFile.is_open()) throw std::runtime_error("Could not open exp ID file");

    std::string line, csvItem;

    while(std::getline(expIDFile, line)){
      std::istringstream myline(line);
      while(getline(myline, csvItem)){
          img_ID_list.push_back(csvItem);
      }
      lineNumber++;
    }
  }

  const size_type num_views = img_ID_list.size();

  FrameTransformList fid_xform_list;
  std::vector<Pt3> Kwire_tip_pt_2d_list, Last_ring_center_pt_2d_list;
  for(size_type idx = 0; idx < num_views; ++idx)
  {
    const std::string img_ID          = img_ID_list[idx];

    // Read drill pnp xform h5 file
    const std::string fid_xform_h5    = fid_xform_folder_path + "/" + carm_pose_xform_prefix + img_ID + ".h5";
    FrameTransform fid_xform          = ReadITKAffineTransformFromFile(fid_xform_h5);
    fid_xform_list.push_back(fid_xform);

    const std::string kwire_tip_annot_txt_path = kwire_annot_folder_path + "/kwire_tip" + img_ID + ".txt";
    Pt3 Kwire_tip_pt_2d = ReadKwireAnnotationPtFromTxt(kwire_tip_annot_txt_path);
    Kwire_tip_pt_2d_list.push_back(Kwire_tip_pt_2d);

    const std::string last_ring_center_annot_txt_path = kwire_annot_folder_path + "/last_ring_center" + img_ID + ".txt";
    Pt3 Last_ring_center_pt_2d = ReadKwireAnnotationPtFromTxt(last_ring_center_annot_txt_path);
    Last_ring_center_pt_2d_list.push_back(Last_ring_center_pt_2d);
  }

  FrameTransform kwire_init_pt_ref = SetRefFrame(kwire_pt_3d);
  FrameTransformList kwire_pt_wrt_cam_list;
  for(size_type idx = 0; idx < num_views; ++idx)
  {
    // Push to kwire_pt_wrt_cam transformation list
    FrameTransform kwire_pt_wrt_cam = kwire_init_pt_ref * fid_xform_list[idx];
    kwire_pt_wrt_cam_list.push_back(kwire_pt_wrt_cam);
  }

  vout << "Brute Force Coarse Searching Kwire tip..." << std::endl;
  Pt3 Kwire_tip_pt_offset_coarse = Brute_Force_Search_Pt(kwire_pt_wrt_cam_list, Kwire_tip_pt_2d_list, num_views, 100, 5);

  Pt3 Searched_Kwire_tip_coarse = kwire_pt_3d + Kwire_tip_pt_offset_coarse;

  vout << "Brute Force Coarse Searching Last ring center..." << std::endl;
  Pt3 Last_ring_center_pt_offset_coarse = Brute_Force_Search_Pt(kwire_pt_wrt_cam_list, Last_ring_center_pt_2d_list, num_views, 100, 5);

  Pt3 Searched_Last_ring_center_coarse = kwire_pt_3d + Last_ring_center_pt_offset_coarse;

  vout << "Brute Force Fine Searching Kwire tip..." << std::endl;
  FrameTransform kwire_tip_coarse_ref = SetRefFrame(Searched_Kwire_tip_coarse);
  kwire_pt_wrt_cam_list.clear();
  for(size_type idx = 0; idx < num_views; ++idx)
  {
    // Push to kwire_pt_wrt_cam transformation list
    FrameTransform kwire_pt_wrt_cam = kwire_tip_coarse_ref * fid_xform_list[idx];
    kwire_pt_wrt_cam_list.push_back(kwire_pt_wrt_cam);
  }
  Pt3 Kwire_tip_pt_offset_fine = Brute_Force_Search_Pt(kwire_pt_wrt_cam_list, Kwire_tip_pt_2d_list, num_views, 10, 0.5);

  Pt3 Searched_Kwire_tip = Searched_Kwire_tip_coarse + Kwire_tip_pt_offset_fine;

  vout << "Reprojecting seached points..." << std::endl;
  for(int idx = 0; idx < num_views; ++idx)
  {
    LandMap3 reproj_kwire_tip_fcsv = ReprojSearchePts(kwire_pt_wrt_cam_list[idx], Kwire_tip_pt_offset_fine);
    WriteFCSVFileFromNamePtMap(output_path + "/reproj_kwire_tip_pt" + img_ID_list[idx] + ".fcsv", reproj_kwire_tip_fcsv);
  }

  vout << "Brute Force Fine Searching Last ring center..." << std::endl;
  FrameTransform Last_ring_center_coarse_ref = SetRefFrame(Searched_Last_ring_center_coarse);
  kwire_pt_wrt_cam_list.clear();
  for(size_type idx = 0; idx < num_views; ++idx)
  {
    // Push to kwire_pt_wrt_cam transformation list
    FrameTransform kwire_pt_wrt_cam = Last_ring_center_coarse_ref * fid_xform_list[idx];
    kwire_pt_wrt_cam_list.push_back(kwire_pt_wrt_cam);
  }
  Pt3 Last_ring_center_pt_offset_fine = Brute_Force_Search_Pt(kwire_pt_wrt_cam_list, Last_ring_center_pt_2d_list, num_views, 10, 0.5);

  Pt3 Searched_Last_ring_center = Searched_Last_ring_center_coarse + Last_ring_center_pt_offset_fine;

  vout << "Reprojecting seached points..." << std::endl;
  for(int idx = 0; idx < num_views; ++idx)
  {
    LandMap3 reproj_last_ring_center_fcsv = ReprojSearchePts(kwire_pt_wrt_cam_list[idx], Last_ring_center_pt_offset_fine);
    WriteFCSVFileFromNamePtMap(output_path + "/reproj_last_ring_center" + img_ID_list[idx] + ".fcsv", reproj_last_ring_center_fcsv);
  }

  LandMap3 triangulated_kwire_pt_fcsv;

  std::pair<std::string, Pt3> Triangulated_Kwire_tip("KwireTip", Searched_Kwire_tip);
  triangulated_kwire_pt_fcsv.insert(Triangulated_Kwire_tip);

  std::pair<std::string, Pt3> Triangulated_Last_ring_center("LastRingCenter", Searched_Last_ring_center);
  triangulated_kwire_pt_fcsv.insert(Triangulated_Last_ring_center);

  ConvertRASToLPS(&triangulated_kwire_pt_fcsv);

  vout << "Saving triangulated points to fcsv..." << std::endl;
  WriteFCSVFileFromNamePtMap(triangulated_fcsv, triangulated_kwire_pt_fcsv);

  return 0;
}
