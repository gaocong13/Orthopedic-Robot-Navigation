
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

  po.set_help("Device pose calculation using PnP solution");
  po.set_arg_usage("<Device 2D landmark annotation ROOT path> <Device 3D landmark annotation FILE path> <Image ID list txt file path> <Image DICOM ROOT path> <Output folder path>");
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

  const std::string device_2d_fcsv_path     = po.pos_args()[0];  // 2D Landmark root path
  const std::string meta_data_path          = po.pos_args()[1];  // Meta Data path
  const std::string exp_list_path           = po.pos_args()[2];  // Experiment list file path
  const std::string dicom_path              = po.pos_args()[3];  // Dicom image path
  const std::string output_path             = po.pos_args()[4];  // Output path

  const std::string device_3d_fcsv_path = meta_data_path + "/Device3Dbb.fcsv";
  std::cout << "reading device BB landmarks from FCSV file..." << std::endl;
  auto device_3d_fcsv = ReadFCSVFileNamePtMap(device_3d_fcsv_path);
  ConvertRASToLPS(&device_3d_fcsv);

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

  const auto default_cam = NaiveCamModelFromCIOSFusion(
                                  MakeNaiveCIOSFusionMetaDR(), true);

  bool is_first_view = true;

  for(int idx=0; idx<lineNumber; ++idx)
  {
    const std::string exp_ID                = exp_ID_list[idx];
    const std::string devicebb_2d_fcsv_path  = device_2d_fcsv_path + "/reproj_bb" + exp_ID + ".fcsv";
    const std::string img_path              = dicom_path + "/" + exp_ID;

    std::cout << "Running..." << exp_ID << std::endl;
    auto devicebb_2d_fcsv = ReadFCSVFileNamePtMap(devicebb_2d_fcsv_path);
    ConvertRASToLPS(&devicebb_2d_fcsv);

    xregASSERT(devicebb_2d_fcsv.size() > 3);

    LandMap2 deviceproj_lands;

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

    FrameTransform pnp_cam_to_device = PnPPOSITAndReprojCMAES(projs_to_regi[0].cam, device_3d_fcsv, projs_to_regi[0].landmarks);

    WriteITKAffineTransform(output_path + "/device_pnp_xform" + exp_ID + ".h5", pnp_cam_to_device);
  }

  return 0;
}
