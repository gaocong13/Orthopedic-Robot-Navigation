
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

  po.set_help("Brute Force search for fiducial point reconstruction. Reproject estimated fiducial point on each 2D image plane and minimize the 2D distance loss.");
  po.set_arg_usage("<Fiducial (such as pelvis) xform folder> <expID list> <Initial Kwire Point FCSV> <output folder> <Kwire Point Name>");
  po.set_min_num_pos_args(5);

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
  const std::string img_ID_list_path             = po.pos_args()[1]; // image ID list path
  const std::string kwire_pt_fcsv_path           = po.pos_args()[2]; // Initial tip position landmark
  const std::string output_path                  = po.pos_args()[3];
  const std::string kwire_pt_name                = po.pos_args()[4];

  std::cout << "reading init device kwire_pt landmark from FCSV file..." << std::endl;
  auto kwire_pt_3dfcsv = ReadFCSVFileNamePtMap(kwire_pt_fcsv_path);
  ConvertRASToLPS(&kwire_pt_3dfcsv);

  FrameTransform kwire_pt_ref = FrameTransform::Identity();
  Pt3 kwire_pt_3d;
  {
    auto kwire_pt_ref_fcsv = kwire_pt_3dfcsv.find(kwire_pt_name);

    if (kwire_pt_ref_fcsv != kwire_pt_3dfcsv.end()){
      kwire_pt_3d = kwire_pt_ref_fcsv->second;
    }
    else{
      std::cout << "ERROR: NOT FOUND KWIRE REF PT" << std::endl;
    }

    kwire_pt_ref.matrix()(0,3) = -kwire_pt_3d[0];
    kwire_pt_ref.matrix()(1,3) = -kwire_pt_3d[1];
    kwire_pt_ref.matrix()(2,3) = -kwire_pt_3d[2];
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

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  FrameTransformList kwire_pt_wrt_cam_list;
  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string img_ID          = img_ID_list[idx];

    // Read drill pnp xform h5 file
    const std::string fid_xform_h5    = fid_xform_folder_path + "/pelvis_sv_regi_xform" + img_ID + ".h5";
    FrameTransform fid_xform          = ReadITKAffineTransformFromFile(fid_xform_h5);

    // Push to kwire_pt_wrt_cam transformation list
    FrameTransform kwire_pt_wrt_cam = kwire_pt_ref * fid_xform;
    kwire_pt_wrt_cam_list.push_back(kwire_pt_wrt_cam);
  }

  // Feb25 2022 ImageList Needle Base
  /*
  // Annotation: 16 tip
  Pt3 Kwire_pt_2d_1 = {1536 - 982, 1536 - 890, 0};
  Pt3 Kwire_pt_2d_2 = {1536 - 794, 1536 - 892, 0};
  Pt3 Kwire_pt_2d_3 = {1536 - 1028, 1536 - 886, 0};
  */

  /*
  // Annotation: 36 tip
  Pt3 Kwire_pt_2d_1 = {1536 - 712, 1536 - 1088, 0};
  Pt3 Kwire_pt_2d_2 = {1536 - 448, 1536 - 1120, 0};
  Pt3 Kwire_pt_2d_3 = {1536 - 754, 1536 - 1128, 0};
  */

  // Annotation: 36 last ring center
  Pt3 Kwire_pt_2d_1 = {1536 - 1002, 1536 - 1246, 0};
  Pt3 Kwire_pt_2d_2 = {1536 - 698, 1536 - 1276, 0};
  Pt3 Kwire_pt_2d_3 = {1536 - 1052, 1536 - 1294, 0};

  std::vector<Pt3> Kwire_pt_2d_list;
  Kwire_pt_2d_list.push_back(Kwire_pt_2d_1);
  Kwire_pt_2d_list.push_back(Kwire_pt_2d_2);
  Kwire_pt_2d_list.push_back(Kwire_pt_2d_3);
  // Kwire_pt_2d_list.push_back(Kwire_pt_2d_4);
  // Kwire_pt_2d_list.push_back(Kwire_pt_2d_5);
  // Kwire_pt_2d_list.push_back(Kwire_pt_2d_6);

  float cur_loss = 0.0;
  float tot_loss = 100000;
  Pt3 offset(3);

  // Perform Searching
  for(float dx = -100; dx < 100; dx += 5)
  {
    for(float dy = -200; dy < 200; dy += 5)
    {
      for(float dz = -100; dz < 100; dz += 5)
      {
        cur_loss = 0.0;
        // Do reprojection and calculate loss
        for(int idx=0; idx<lineNumber; ++idx)
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
          std::cout << "totloss:" << tot_loss << " dx:" << dx << " dy:" << dy << " dz:" << dz << std::endl;
        }
      }
    }

  }

  Pt3 SearchedTip = kwire_pt_3d + offset;

  for( int idx=0; idx<lineNumber; ++idx )
  {
    LandMap3 reproj_bbs_fcsv;
    auto reproj_bb = default_cam.phys_pt_to_ind_pt(kwire_pt_wrt_cam_list[idx].inverse() * offset);
    reproj_bb[0] = 0.194 * (reproj_bb[0] - 1536);
    reproj_bb[1] = 0.194 * (reproj_bb[1] - 1536);
    reproj_bb[2] = 0;
    std::pair<std::string, Pt3> ld2_3D("Searched Reprojection", reproj_bb);
    reproj_bbs_fcsv.insert(ld2_3D);
    WriteFCSVFileFromNamePtMap(output_path + "/reproj_kwire_pt" + img_ID_list[idx] + ".fcsv", reproj_bbs_fcsv);
  }

  std::pair<std::string, Pt3> NeedleTip_3D("LastRingCenter", SearchedTip);
  LandMap3 calibrated_kwire_pt_fcsv;
  calibrated_kwire_pt_fcsv.insert(NeedleTip_3D);

  ConvertRASToLPS(&calibrated_kwire_pt_fcsv);
  WriteFCSVFileFromNamePtMap(output_path + "/CalibratedKwirePt_lastringcenter.fcsv", calibrated_kwire_pt_fcsv);

  return 0;
}
