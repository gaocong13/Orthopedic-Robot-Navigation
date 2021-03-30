
// Define this if you have highgui and want the debug images displayed
// as part of the radial symmetry function
//#define JHMR_HAS_OPENCV_HIGHGUI

#include "CommonTBME.h"

// Boost
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/null.hpp>
#include <boost/any.hpp>

// For the BB detection 
#include <opencv2/opencv.hpp>
#ifdef JHMR_HAS_OPENCV_HIGHGUI
#include <opencv2/highgui/highgui.hpp>
#endif

#include "jhmrAnatCoordFrames.h"
#include "jhmrHounsfieldToLinearAttenuationFilter.h"

#include "jhmrGPUPrefsXML.h"

#include "jhmrRegi2D3DCMAES.h"
#include "jhmrRegi2D3DBOBYQA.h"
#include "jhmrRegi2D3DIntensityExhaustive.h"
#include "jhmrGradientNCCImageSimilarityMetricGPU.h"
#include "jhmrPatchGradNCCImageSimilarityMetricGPU.h"

#include "jhmrFoldedNormalDist.h"
#include "jhmrNormalDist.h"
#include "jhmrRegi2D3DSE3MagPenaltyFn.h"
#include "jhmrRegi2D3DSE3EulerDecompPenaltyFn.h"

#include "jhmrProjDataIO.h"
#include "jhmrPAOCutsXML.h"
#include "jhmrFCSVUtils.h"
#include "jhmrCIOSFusionDICOM.h"

#include "jhmrRigidRegiUtils.h"
#include "jhmrMultipleViewUtils.h"
#include "jhmrSpatialPrimitives.h"
#include "jhmrSegBB.h"

namespace jhmr
{
namespace tbme
{

bool& NeedToSetupViennaCL()
{
  static bool n = true;
  return n;
}

}  // tbme
}  // jhmr

namespace
{

using namespace jhmr;
using namespace jhmr::tbme;

template <class tSimMetric>
void MakeGPUSimMetricDetails(tSimMetric* sm, const FragRegiParams& params, const size_type lvl)
{

}

template <>
void MakeGPUSimMetricDetails<PatchGradNCCImageSimilarityMetricGPU>(PatchGradNCCImageSimilarityMetricGPU* sm,
                                                                   const FragRegiParams& params,
                                                                   const size_type lvl)
{
  if (false)
  {
    double ds_factor = 0;

    if (lvl == 0)
    {
      ds_factor = params.lvl_1_ds_factor;
    }
    else
    {
      jhmrASSERT(lvl == 1);
      ds_factor = params.lvl_2_ds_factor;
    }
   
    const size_type patch_radius_full_res = 41;
    //const size_type patch_radius_full_res = 81;

    size_type patch_radius = std::lround(ds_factor * patch_radius_full_res);

    sm->set_patch_radius(patch_radius);
    
    //sm->set_patch_stride(patch_radius);
    //sm->set_patch_stride((2 * patch_radius) + 1);
    sm->set_patch_stride(1);
  }
}

template <class tSimMetric>
std::shared_ptr<MultiLevelMultiObjRegi::SimMetric>
MakeGPUSimMetric(const boost::compute::context& ctx, const boost::compute::command_queue& queue,
                 const FragRegiParams& params, const size_type lvl)
{
  auto sm = std::make_shared<tSimMetric>(ctx, queue);
      
  if (lvl == 0)
  {
    if (NeedToSetupViennaCL())
    {
      sm->set_setup_vienna_cl_ctx(true);
      NeedToSetupViennaCL() = false;
    }
    else
    {
      sm->set_setup_vienna_cl_ctx(false);
    }
    
    if (params.lvl_1_smooth_kernel_size)
    {
      sm->set_smooth_img_before_sobel_kernel_radius(*params.lvl_1_smooth_kernel_size);
    }
  }
  else if (lvl == 1)
  {
    if (params.lvl_2_smooth_kernel_size)
    {
      sm->set_smooth_img_before_sobel_kernel_radius(*params.lvl_2_smooth_kernel_size);
    }

    sm->set_setup_vienna_cl_ctx(false);
  }

  MakeGPUSimMetricDetails(sm.get(), params, lvl);

  return sm;
};

}  // un-named

std::ostream& jhmr::tbme::CommonProgArgs::vout()
{
  if (!verbose)
  {
    if (!null_vout)
    {
      null_vout = std::make_shared<boost::iostreams::stream<boost::iostreams::null_sink>>((boost::iostreams::null_sink()));
    }
    return *null_vout;
  }
  else
  {
    return std::cout;
  }
}

void jhmr::tbme::SetupCommonProgArgs(ProgOpts* po, const CommonProgArgs& args)
{
  po->add("name", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_STRING, "name",
         "Name of the volume and projection groups to run registration against.")
    << args.data_name;

  po->add("single-cut", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "single-cut",
          "Only run over a single random cut index, when not provided, all cuts are run.");

  po->add("single-repo", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "single-repo",
          "Only run over a single fragment reposition index, when not provided, all repositions are run.");

  po->add("single-init", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "single-init",
          "Only run over a single registration initialization, when not provided, all initializations are run.");

  po->add_tbb_max_num_threads_flag();

  po->add("tmp-root", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_STRING, "tmp-root",
          "Root directory used to to create more temporary directories for work.")
    << args.tmp_dir_root;

  po->add("verbose", 'v', ProgOpts::kSTORE_TRUE, "verbose",
          "Print verbose information to stdout.")
    << false;
}

jhmr::tbme::CommonProgArgs jhmr::tbme::GetCommonProgArgs(const ProgOpts& po)
{
  CommonProgArgs args;

  args.verbose = po.get("verbose");

  args.data_name = po.get("name").as_string();

  if (po.has("single-cut"))
  {
    args.single_cut_idx = po.get("single-cut").as_uint32();
  }

  if (po.has("single-repo"))
  {
    args.single_repo_idx = po.get("single-repo").as_uint32();
  }

  if (po.has("single-init"))
  {
    args.single_init_idx = po.get("single-init").as_uint32();
  }

  args.tmp_dir_root = po.get("tmp-root").as_string();

  return args;
}

void jhmr::tbme::SetupMaskProgArgs(ProgOpts* po, const std::string& default_name)
{
  po->add("mask-2d", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_STRING, "mask-2d",
          "Name of a previously computed 2D mask to be used during processing. "
          "\"\" (empty string) --> no mask. "
          "\"auto\" --> dumb automatic masking. "
          "\"gt\" --> ground truth masks. "
          "\"manual\" --> manual masks for the cadaver studies.")
    << default_name;
}

std::string jhmr::tbme::GetMaskProgArgs(const ProgOpts& po)
{
  return po.get("mask-2d").as_string();
}

std::string jhmr::tbme::GetFragRegiNameUsingMaskStr(const std::string frag_regi_str, const std::string& mask_arg)
{
  std::string mask_str_suffix;

  if (mask_arg == "auto")
  {
    mask_str_suffix = "-mask";
  }
  else if (mask_arg == "gt")
  {
    mask_str_suffix = "-gt-mask";
  }
  else
  {
    jhmrASSERT(mask_arg.empty());
  }

  return fmt::sprintf("%s%s", frag_regi_str, mask_str_suffix);
}

std::string jhmr::tbme::GetMaskProjGroupName(const std::string mask_arg)
{
  jhmrASSERT(!mask_arg.empty());

  if (mask_arg == "full-seg-gt")
  {
    return "gt-full-seg-projs";
  }
  else
  {
    return fmt::sprintf("%s-mask-projs", mask_arg);
  }
}

jhmr::size_type jhmr::tbme::MapPatientIDToIndex(const std::string& pat_id)
{
  static bool need_to_init_map = true;

  static std::unordered_map<std::string,size_type> m;

  if (need_to_init_map)
  {
    m = { { "17-1905", 1 },
          { "17-1882", 2 },
          { "18-1109", 3 },
          { "18-0725", 4 },
          { "18-2799", 5 },
          { "18-2800", 6 } };

    need_to_init_map = false;
  }

  return m.find(pat_id)->second;
}

std::string jhmr::tbme::GetPatientID(const H5::CommonFG& h5)
{
  return ReadStringH5("specimen-id", h5);
}

std::string jhmr::tbme::GetSideString(const H5::CommonFG& h5)
{
  return ReadStringH5("side", h5);
}

bool jhmr::tbme::IsLeftSide(const H5::CommonFG& h5)
{
  const std::string side_str = GetSideString(h5);
  
  const bool is_left = side_str == "left";

  jhmrASSERT(is_left || (side_str == "right"));

  return is_left;
}

std::string jhmr::tbme::GetAnatLandmarksMapH5Path()
{
  return "anat-landmarks";
}

jhmr::LandMap3
jhmr::tbme::GetAnatLandmarksMap(const H5::CommonFG& h5)
{
  return ReadLandmarksMapH5<Pt3>(h5.openGroup(GetAnatLandmarksMapH5Path()));
}

jhmr::FrameTransform jhmr::tbme::GetAPPToVolStdOrigin(const H5::CommonFG& h5)
{
  const auto anat_lands = GetAnatLandmarksMap(h5);

  FrameTransform app_to_vol;

  AnteriorPelvicPlaneFromLandmarksMap(anat_lands, &app_to_vol, kAPP_ORIGIN_MEDIAN_ILIAC_CREST);

  return app_to_vol;
}

jhmr::FrameTransform jhmr::tbme::GetAPPToVolFHOrigin(const H5::CommonFG& h5)
{
  const auto anat_lands = GetAnatLandmarksMap(h5);

  FrameTransform app_to_vol;

  AnteriorPelvicPlaneFromLandmarksMap(anat_lands, &app_to_vol,
                                      IsLeftSide(h5) ? kAPP_ORIGIN_LEFT_FH : kAPP_ORIGIN_RIGHT_FH);

  return app_to_vol;
}

std::array<jhmr::Pt3,2> jhmr::tbme::GetBothFHWrtVol(const H5::CommonFG& h5)
{
  const auto anat_lands = GetAnatLandmarksMap(h5);
 
  const auto left_fh_it = anat_lands.find("FH-l");
  jhmrASSERT(left_fh_it != anat_lands.end());
 
  const auto right_fh_it = anat_lands.find("FH-r");
  jhmrASSERT(right_fh_it != anat_lands.end());

  return { left_fh_it->second, right_fh_it->second };
}

jhmr::Pt3 jhmr::tbme::GetIpsilFHPtWrtVol(const H5::CommonFG& h5)
{
  const auto fhs = GetBothFHWrtVol(h5);

  return IsLeftSide(h5) ? fhs[0] : fhs[1];
}

jhmr::Pt3 jhmr::tbme::GetFHMidPtWrtVol(const H5::CommonFG& h5)
{
  const auto fhs = GetBothFHWrtVol(h5);
  
  return (fhs[0] + fhs[1]) / 2;
}

jhmr::tbme::PelvisFragFemurLabels
jhmr::tbme::GetPelvisFragFemurLabels(const H5::CommonFG& h5, const LabelVol* seg)
{
  const auto fhs = GetBothFHWrtVol(h5);

  const bool is_left = IsLeftSide(h5);

  const auto ipsil_fh  = is_left ? fhs[0] : fhs[1];
  const auto contra_fh = is_left ? fhs[1] : fhs[0];

  PelvisFragFemurLabels anat_labels;

  PAOGuessPelvisFemurFragLabels(seg,
                                &anat_labels.pelvis_label,
                                &anat_labels.femur_label,
                                &anat_labels.frag_label,
                                &ipsil_fh);

  // get the contralateral femur label
  
  LabelVol::IndexType contra_idx;
  
  LabelVol::PointType contra_itk_pt;
  contra_itk_pt[0] = contra_fh[0];
  contra_itk_pt[1] = contra_fh[1];
  contra_itk_pt[2] = contra_fh[2];

  seg->TransformPhysicalPointToIndex(contra_itk_pt, contra_idx);

  anat_labels.contra_femur_label = seg->GetPixel(contra_idx);

  return anat_labels;
}

std::string jhmr::tbme::GetOrigIntensVolH5Path()
{
  return "vol";
}

std::string jhmr::tbme::GetPelvisFemursSegH5Path()
{
  return "vol-seg";
}

std::string jhmr::tbme::GetGTCutsSegH5Path()
{
  return "gt-cuts-seg";
}

std::string jhmr::tbme::GetGTCutsDefH5Path()
{
  return "gt-cuts-fit-plan";
}

std::string jhmr::tbme::GetActualCutsSegH5Path(const size_type cut_idx)
{
  return fmt::sprintf("rand-plans/%02lu/seg", cut_idx);
}

std::string jhmr::tbme::GetActualCutsDefH5Path(const size_type cut_idx)
{
  return fmt::sprintf("rand-plans/%02lu/def", cut_idx);
}

std::string jhmr::tbme::GetKnownPubisIliumCutsSegH5Path(const size_type cut_idx)
{
  return fmt::sprintf("rand-plans/%02lu/seg-known-pu-il", cut_idx);
}

std::string jhmr::tbme::GetProjDataPath(const size_type cut_idx,
                                        const size_type repo_idx,
                                        const std::string& data_name)
{
  return fmt::sprintf("rand-plans/%02lu/repos/%02lu/projs/%s",
                      cut_idx, repo_idx, data_name);
}

jhmr::VolPtr jhmr::tbme::GetOrigIntensVol(const H5::CommonFG& h5)
{
  return ReadITKImageH5<PixelScalar,3>(h5.openGroup(GetOrigIntensVolH5Path()));
}

jhmr::VolPtr jhmr::tbme::GetOrigIntensVolAsLinAtt(const H5::CommonFG& h5)
{
  VolPtr vol = GetOrigIntensVol(h5);
  
  using HU2Att = HounsfieldToLinearAttenuationFilter<Vol>;

  HU2Att::Pointer hu2att = HU2Att::New();
  hu2att->SetInput(vol);
  hu2att->Update();

  return hu2att->GetOutput();
}

jhmr::LabelVolPtr jhmr::tbme::GetPelvisSeg(const H5::CommonFG& h5)
{
  return ReadITKImageH5<LabelScalar,3>(h5.openGroup(GetPelvisFemursSegH5Path()));
}

jhmr::LabelVolPtr jhmr::tbme::GetGTCutsSeg(const H5::CommonFG& h5)
{
  return ReadITKImageH5<LabelScalar,3>(h5.openGroup(GetGTCutsSegH5Path()));
}

H5::Group jhmr::tbme::GetRandPlansParentGroup(const H5::CommonFG& h5)
{
  return h5.openGroup("rand-plans");
}

H5::Group jhmr::tbme::GetReposParentGroup(const H5::CommonFG& h5)
{
  return h5.openGroup("repos");
}

std::vector<H5::Group> jhmr::tbme::GetNumberedGroups(const H5::CommonFG& h5)
{
  std::vector<H5::Group> gs;
  gs.reserve(20);

  HideH5ExceptionPrints h;

  bool has_more = true;
  for (size_type i = 0; has_more; ++i)
  {
    try
    {
      gs.push_back(h5.openGroup(fmt::sprintf("%02lu", i)));
    }
    catch (...)
    {
      has_more = false;
    }
  }

  return gs;
}

std::vector<H5::Group>
jhmr::tbme::GetSubsetOfOneNumberedGroup(const std::vector<H5::Group>& gs, const size_type idx)
{
  return { gs[idx] };
}

std::vector<jhmr::FrameTransform>
jhmr::tbme::GetInitCamWrtVols(const H5::CommonFG& h5)
{
  std::vector<FrameTransform> xforms;
  xforms.reserve(5);
  
  HideH5ExceptionPrints h;
  
  bool has_more = true;
  for (size_type i = 0; has_more; ++i)
  {
    try
    {
      xforms.push_back(ReadAffineTransform4x4H5<CoordScalar>(fmt::sprintf("init-cam-wrt-vol-%02lu", i), h5));
    }
    catch (...)
    {
      has_more = false;
    }
  }
  
  return xforms;
}

namespace
{

using namespace jhmr;
using namespace jhmr::tbme;

std::tuple<size_type,size_type,size_type>
LookupCustomPopSizes(const size_type default_pop_size_pelvis,
                     const size_type default_pop_size_femur,
                     const size_type default_pop_size_frag)
{
  size_type pelvis_pop_size = default_pop_size_pelvis;
  size_type femur_pop_size  = default_pop_size_femur;
  size_type frag_pop_size   = default_pop_size_frag;
  
  const auto& m = GlobalDebugDataManager();

  const auto pelvis_pop_it = m.find("pop-size-pelvis");
  if (pelvis_pop_it != m.end())
  {
    pelvis_pop_size = boost::any_cast<size_type>(pelvis_pop_it->second);
  }

  const auto femur_pop_it = m.find("pop-size-femur");
  if (femur_pop_it != m.end())
  {
    femur_pop_size = boost::any_cast<size_type>(femur_pop_it->second);
  }

  const auto frag_pop_it = m.find("pop-size-frag");
  if (frag_pop_it != m.end())
  {
    frag_pop_size = boost::any_cast<size_type>(frag_pop_it->second);
  }

  return std::make_tuple(pelvis_pop_size, femur_pop_size, frag_pop_size);
}

}  // un-named

jhmr::tbme::FragRegiParams jhmr::tbme::MakeFragRegiParamsOldSim(const bool use_patch_sim)
{
  FragRegiParams p;
  
  if (use_patch_sim)
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<PatchGradNCCImageSimilarityMetricGPU>;
  }
  else
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<GradientNCCImageSimilarityMetric2DGPU>;
  }

  const auto pop_sizes = LookupCustomPopSizes(100, 100, 100);

  p.lvl_1_ds_factor = 0.125;

  p.lvl_1_pelvis_pop_size = std::get<0>(pop_sizes);
  p.lvl_1_pelvis_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_pelvis_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 100, 100, 100 };
  p.lvl_1_pelvis_tol_fn = 0.01;
  p.lvl_1_pelvis_tol_x = 0.01;

  p.lvl_1_femur_pop_size = std::get<1>(pop_sizes);
  p.lvl_1_femur_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_femur_bounds = { 20 * kDEG2RAD, 20 * kDEG2RAD, 20 * kDEG2RAD, 100, 100, 100 };
  p.lvl_1_femur_tol_fn = 0.01;
  p.lvl_1_femur_tol_x = 0.01;

  p.lvl_1_frag_pop_size = std::get<2>(pop_sizes);
  p.lvl_1_frag_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_frag_bounds = { 20 * kDEG2RAD, 20 * kDEG2RAD, 20 * kDEG2RAD, 100, 100, 100 };
  p.lvl_1_frag_tol_fn = 0.01;
  p.lvl_1_frag_tol_x = 0.01;

  p.lvl_2_ds_factor = 0.25;
  
  p.lvl_2_pelvis_bounds = { 5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_pelvis_tol_fn = 0.001;
  p.lvl_2_pelvis_tol_x = 0.001;

  p.lvl_2_femur_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_femur_tol_fn = 0.001;
  p.lvl_2_femur_tol_x = 0.001;

  p.lvl_2_frag_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_frag_tol_fn = 0.001;
  p.lvl_2_frag_tol_x = 0.001;

  p.lvl_2_all_bounds = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10  };
  p.lvl_2_all_tol_fn = 0.001;
  p.lvl_2_all_tol_x = 0.001;

  return p;
}

jhmr::tbme::FragRegiParams jhmr::tbme::MakeFragRegiParamsNewSim(const bool use_patch_sim)
{
  FragRegiParams p;
 
  if (use_patch_sim)
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<PatchGradNCCImageSimilarityMetricGPU>;
  }
  else
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<GradientNCCImageSimilarityMetric2DGPU>;
  }
  
  const auto pop_sizes = LookupCustomPopSizes(100, 100, 100);

  p.lvl_1_ds_factor = 0.125;

  p.lvl_1_smooth_kernel_size = 5;

  p.lvl_1_pelvis_pop_size = std::get<0>(pop_sizes);
  p.lvl_1_pelvis_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_pelvis_bounds = { 20 * kDEG2RAD, 20 * kDEG2RAD, 20 * kDEG2RAD, 100, 100, 100 };
  p.lvl_1_pelvis_tol_fn = 0.01;
  p.lvl_1_pelvis_tol_x = 0.01;

  p.lvl_1_femur_pop_size = std::get<1>(pop_sizes);
  p.lvl_1_femur_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_femur_bounds = { 0.7, 0.7, 0.7, 100, 100, 100 };  // 0.7 rad ~ 40.1 deg
  p.lvl_1_femur_tol_fn = 0.01;
  p.lvl_1_femur_tol_x = 0.01;

  p.lvl_1_frag_pop_size = std::get<2>(pop_sizes);
  p.lvl_1_frag_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_frag_bounds = { 0.7, 0.7, 0.7, 100, 100, 100 };  // 0.7 rad ~ 40.1 deg
  p.lvl_1_frag_tol_fn = 0.01;
  p.lvl_1_frag_tol_x = 0.01;

  p.lvl_2_ds_factor = 0.25;
  
  p.lvl_2_smooth_kernel_size = 11;
  
  p.lvl_2_pelvis_bounds = { 5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_pelvis_tol_fn = 0.001;
  p.lvl_2_pelvis_tol_x = 0.001;

  p.lvl_2_femur_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_femur_tol_fn = 0.001;
  p.lvl_2_femur_tol_x = 0.001;

  p.lvl_2_frag_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_frag_tol_fn = 0.001;
  p.lvl_2_frag_tol_x = 0.001;

  p.lvl_2_all_bounds = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10  };
  p.lvl_2_all_tol_fn = 0.001;
  p.lvl_2_all_tol_x = 0.001;

  return p;
}

jhmr::tbme::FragRegiParams jhmr::tbme::MakeFragRegiParamsNewSimPelvisFid(const bool use_patch_sim)
{
  FragRegiParams p;
  
  if (use_patch_sim)
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<PatchGradNCCImageSimilarityMetricGPU>;
  }
  else
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<GradientNCCImageSimilarityMetric2DGPU>;
  }
  
  const auto pop_sizes = LookupCustomPopSizes(20, 100, 100);

  p.lvl_1_ds_factor = 0.125;

  p.lvl_1_smooth_kernel_size = 5;

  p.lvl_1_pelvis_pop_size = std::get<0>(pop_sizes);
  p.lvl_1_pelvis_sigma = { 0.04, 0.04, 0.04, 2, 2, 2 };
  p.lvl_1_pelvis_bounds = { 5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 10, 10, 10 };
  p.lvl_1_pelvis_tol_fn = 0.01;
  p.lvl_1_pelvis_tol_x = 0.01;

  p.lvl_1_femur_pop_size = std::get<1>(pop_sizes);
  p.lvl_1_femur_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_femur_bounds = { 0.7, 0.7, 0.7, 100, 100, 100 };  // 0.7 rad ~ 40.1 deg
  p.lvl_1_femur_tol_fn = 0.01;
  p.lvl_1_femur_tol_x = 0.01;

  p.lvl_1_frag_pop_size = std::get<2>(pop_sizes);
  p.lvl_1_frag_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_frag_bounds = { 0.7, 0.7, 0.7, 100, 100, 100 };  // 0.7 rad ~ 40.1 deg
  p.lvl_1_frag_tol_fn = 0.01;
  p.lvl_1_frag_tol_x = 0.01;

  p.lvl_2_ds_factor = 0.25;
  
  p.lvl_2_smooth_kernel_size = 11;
  
  p.lvl_2_pelvis_bounds = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 5, 5, 5 };
  p.lvl_2_pelvis_tol_fn = 0.001;
  p.lvl_2_pelvis_tol_x = 0.001;

  p.lvl_2_femur_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_femur_tol_fn = 0.001;
  p.lvl_2_femur_tol_x = 0.001;

  p.lvl_2_frag_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_frag_tol_fn = 0.001;
  p.lvl_2_frag_tol_x = 0.001;

  p.lvl_2_all_bounds = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 5, 5, 5,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10  };
  p.lvl_2_all_tol_fn = 0.001;
  p.lvl_2_all_tol_x = 0.001;

  return p;
}

jhmr::tbme::FragRegiParams jhmr::tbme::MakeFragRegiParamsNewSimReg(const bool use_patch_sim)
{
  using FoldNormPDF = FoldedNormalPDF<CoordScalar>;
  using NormPDF     = NormalDist1D<CoordScalar>;

  using RotMagAndTransMagPen = Regi2D3DSE3MagPenaltyFn<FoldNormPDF>;

  using EulerRotAndTransCompPen = Regi2D3DSE3EulerDecompPenaltyFn<NormPDF>;
  
  const auto pop_sizes = LookupCustomPopSizes(100, 100, 100);

  FragRegiParams p;

  if (use_patch_sim)
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<PatchGradNCCImageSimilarityMetricGPU>;
  }
  else
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<GradientNCCImageSimilarityMetric2DGPU>;
  }

  p.lvl_1_ds_factor = 0.125;

  p.lvl_1_smooth_kernel_size = 5;

  p.lvl_1_pelvis_pop_size = std::get<0>(pop_sizes);
  p.lvl_1_pelvis_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_pelvis_tol_fn = 0.01;
  p.lvl_1_pelvis_tol_x = 0.01;

  {
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();
    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(10 * kDEG2RAD, 10 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(50, 50));
    
    p.lvl_1_pelvis_reg_fn = pen_fn;
    p.lvl_1_pelvis_reg_img_coeff = 0.9;
  }

  p.lvl_1_femur_pop_size = std::get<1>(pop_sizes);
  p.lvl_1_femur_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_femur_tol_fn = 0.01;
  p.lvl_1_femur_tol_x = 0.01;
  
  {
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();
    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(45 * kDEG2RAD, 45 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(10, 10));
    
    p.lvl_1_femur_reg_fn = pen_fn;
    p.lvl_1_femur_reg_img_coeff = 0.9;
  }

  p.lvl_1_frag_pop_size = std::get<2>(pop_sizes);
  p.lvl_1_frag_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_frag_tol_fn = 0.01;
  p.lvl_1_frag_tol_x = 0.01;
  
  {
#if 0 
    auto pen_fn = std::make_shared<EulerRotAndTransCompPen>();
    
    pen_fn->rot_x_pdf = NormPDF(10 * kDEG2RAD, 7.5 * kDEG2RAD);
    pen_fn->rot_y_pdf = NormPDF( 0 * kDEG2RAD,   5 * kDEG2RAD);
    pen_fn->rot_z_pdf = NormPDF(15 * kDEG2RAD, 7.5 * kDEG2RAD);  // TODO: update sign for left/right negative for left, positive for right
    
    pen_fn->trans_x_pdf = NormPDF(0,7.5);
    pen_fn->trans_y_pdf = NormPDF(0,5);
    pen_fn->trans_z_pdf = NormPDF(0,7.5);
#else
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();

    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(25 * kDEG2RAD, 25 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(10, 10));
#endif
    
    p.lvl_1_frag_reg_fn = pen_fn;
    p.lvl_1_frag_reg_img_coeff = 0.9;
  }

  p.lvl_2_ds_factor = 0.25;
  
  p.lvl_2_smooth_kernel_size = 11;
  
  p.lvl_2_pelvis_bounds = { 5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_pelvis_tol_fn = 0.001;
  p.lvl_2_pelvis_tol_x = 0.001;

  p.lvl_2_femur_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_femur_tol_fn = 0.001;
  p.lvl_2_femur_tol_x = 0.001;

  p.lvl_2_frag_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_frag_tol_fn = 0.001;
  p.lvl_2_frag_tol_x = 0.001;

  p.lvl_2_all_bounds = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10  };
  p.lvl_2_all_tol_fn = 0.001;
  p.lvl_2_all_tol_x = 0.001;

  return p;

}

jhmr::tbme::FragRegiParams jhmr::tbme::MakeFragRegiParamsNewSimPelvisFidReg(const bool use_patch_sim)
{
  using FoldNormPDF = FoldedNormalPDF<CoordScalar>;
  using NormPDF     = NormalDist1D<CoordScalar>;

  using RotMagAndTransMagPen = Regi2D3DSE3MagPenaltyFn<FoldNormPDF>;

  using EulerRotAndTransCompPen = Regi2D3DSE3EulerDecompPenaltyFn<NormPDF>;
  
  const auto pop_sizes = LookupCustomPopSizes(20, 100, 100);
  
  FragRegiParams p;
  
  if (use_patch_sim)
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<PatchGradNCCImageSimilarityMetricGPU>;
  }
  else
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<GradientNCCImageSimilarityMetric2DGPU>;
  }

  p.lvl_1_ds_factor = 0.125;

  p.lvl_1_smooth_kernel_size = 5;

  p.lvl_1_pelvis_pop_size = std::get<0>(pop_sizes);
  p.lvl_1_pelvis_sigma = { 0.04, 0.04, 0.04, 2, 2, 2 };
  p.lvl_1_pelvis_tol_fn = 0.01;
  p.lvl_1_pelvis_tol_x = 0.01;

  {
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();
    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(2.5 * kDEG2RAD, 2.5 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(5, 5));
    
    p.lvl_1_pelvis_reg_fn = pen_fn;
    p.lvl_1_pelvis_reg_img_coeff = 0.9;
  }
  
  p.lvl_1_femur_pop_size = std::get<1>(pop_sizes);
  p.lvl_1_femur_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_femur_tol_fn = 0.01;
  p.lvl_1_femur_tol_x = 0.01;
  
  {
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();
    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(45 * kDEG2RAD, 45 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(10, 10));
    
    p.lvl_1_femur_reg_fn = pen_fn;
    p.lvl_1_femur_reg_img_coeff = 0.9;
  }

  p.lvl_1_frag_pop_size = std::get<2>(pop_sizes);
  p.lvl_1_frag_sigma = { 0.3, 0.3, 0.3, 5, 5, 5 };
  p.lvl_1_frag_tol_fn = 0.01;
  p.lvl_1_frag_tol_x = 0.01;
  
  {
#if 0 
    auto pen_fn = std::make_shared<EulerRotAndTransCompPen>();
    
    pen_fn->rot_x_pdf = NormPDF(10 * kDEG2RAD, 7.5 * kDEG2RAD);
    pen_fn->rot_y_pdf = NormPDF( 0 * kDEG2RAD,   5 * kDEG2RAD);
    pen_fn->rot_z_pdf = NormPDF(15 * kDEG2RAD, 7.5 * kDEG2RAD);  // TODO: update sign for left/right negative for left, positive for right
    
    pen_fn->trans_x_pdf = NormPDF(0,7.5);
    pen_fn->trans_y_pdf = NormPDF(0,5);
    pen_fn->trans_z_pdf = NormPDF(0,7.5);
#else
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();

    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(25 * kDEG2RAD, 25 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(10, 10));
#endif
    
    p.lvl_1_frag_reg_fn = pen_fn;
    p.lvl_1_frag_reg_img_coeff = 0.9;
  }

  p.lvl_2_ds_factor = 0.25;
  
  p.lvl_2_smooth_kernel_size = 11;
  
  p.lvl_2_pelvis_bounds = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 5, 5, 5 };
  p.lvl_2_pelvis_tol_fn = 0.001;
  p.lvl_2_pelvis_tol_x = 0.001;

  p.lvl_2_femur_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_femur_tol_fn = 0.001;
  p.lvl_2_femur_tol_x = 0.001;

  p.lvl_2_frag_bounds = { 10 * kDEG2RAD, 10 * kDEG2RAD, 10 * kDEG2RAD, 30, 30, 30 };
  p.lvl_2_frag_tol_fn = 0.001;
  p.lvl_2_frag_tol_x = 0.001;

  p.lvl_2_all_bounds = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 5, 5, 5,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10,
                         2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 10, 10, 10  };
  p.lvl_2_all_tol_fn = 0.001;
  p.lvl_2_all_tol_x = 0.001;

  return p;
}

jhmr::tbme::FragRegiParams jhmr::tbme::MakePelvisRegiParamsRegAPView(const bool use_patch_sim, const size_type pop_size)
{
  using FoldNormPDF             = FoldedNormalPDF<CoordScalar>;
  using RotMagAndTransMagPen    = Regi2D3DSE3MagPenaltyFn<FoldNormPDF>;
  using NormPDF                 = NormalDist1D<CoordScalar>;
  using EulerRotAndTransCompPen = Regi2D3DSE3EulerDecompPenaltyFn<NormPDF>;

  FragRegiParams p;

  if (use_patch_sim)
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<PatchGradNCCImageSimilarityMetricGPU>;
  }
  else
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<GradientNCCImageSimilarityMetric2DGPU>;
  }

  p.lvl_1_ds_factor = 0.125;

  p.lvl_1_smooth_kernel_size = 5;

  p.lvl_1_pelvis_pop_size = pop_size;
  p.lvl_1_pelvis_sigma = { 15 * kDEG2RAD, 15 * kDEG2RAD, 30 * kDEG2RAD, 50, 50, 100 };
  p.lvl_1_pelvis_tol_fn = 0.01;
  p.lvl_1_pelvis_tol_x = 0.01;

  if (true)
  {
#if 1
    auto pen_fn = std::make_shared<EulerRotAndTransCompPen>();
    pen_fn->rot_x_pdf   = NormPDF(0, 10 * kDEG2RAD);
    pen_fn->rot_y_pdf   = NormPDF(0, 10 * kDEG2RAD);
    pen_fn->rot_z_pdf   = NormPDF(0, 10 * kDEG2RAD);
    pen_fn->trans_x_pdf = NormPDF(0, 20);
    pen_fn->trans_y_pdf = NormPDF(0, 20);
    pen_fn->trans_z_pdf = NormPDF(0, 100);
#else
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();
    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(10 * kDEG2RAD, 10 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(50, 50));
#endif

    p.lvl_1_pelvis_reg_fn = pen_fn;
    p.lvl_1_pelvis_reg_img_coeff = 0.9;
  }
  else
  {
    p.lvl_1_pelvis_bounds = { 30 * kDEG2RAD, 30 * kDEG2RAD, 60 * kDEG2RAD, 100, 100, 200 };
  }
  
  p.lvl_2_ds_factor = 0.25;
  
  p.lvl_2_smooth_kernel_size = 11;
  
  {
    const double ds_mult = p.lvl_1_ds_factor * (1.0 / p.lvl_2_ds_factor / 2.0);

    p.lvl_2_pelvis_bounds = { ds_mult * 30 * kDEG2RAD, ds_mult * 30 * kDEG2RAD, ds_mult * 60 * kDEG2RAD,
                              ds_mult * 100, ds_mult * 100, ds_mult * 200 };
  }

  p.lvl_2_pelvis_tol_fn = 0.0001;
  p.lvl_2_pelvis_tol_x = 0.0001;

  return p;
}

jhmr::tbme::FragRegiParams jhmr::tbme::MakePelvisRegiParamsRegRotFromAPView(const bool use_patch_sim, const size_type pop_size)
{
  using FoldNormPDF             = FoldedNormalPDF<CoordScalar>;
  using RotMagAndTransMagPen    = Regi2D3DSE3MagPenaltyFn<FoldNormPDF>;
  using NormPDF                 = NormalDist1D<CoordScalar>;
  using EulerRotAndTransCompPen = Regi2D3DSE3EulerDecompPenaltyFn<NormPDF>;

  FragRegiParams p;

  if (use_patch_sim)
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<PatchGradNCCImageSimilarityMetricGPU>;
  }
  else
  {
    p.get_sim_metric_obj = &MakeGPUSimMetric<GradientNCCImageSimilarityMetric2DGPU>;
  }

  p.lvl_1_ds_factor = 0.125;

  p.lvl_1_smooth_kernel_size = 5;

  p.lvl_1_pelvis_pop_size = pop_size;
  p.lvl_1_pelvis_sigma = { 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5 * kDEG2RAD, 2.5, 2.5, 25 };
  p.lvl_1_pelvis_tol_fn = 0.01;
  p.lvl_1_pelvis_tol_x = 0.01;

  if (true)
  {
#if 1
    auto pen_fn = std::make_shared<EulerRotAndTransCompPen>();
    pen_fn->rot_x_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
    pen_fn->rot_y_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
    pen_fn->rot_z_pdf   = NormPDF(0, 2.5 * kDEG2RAD);
    pen_fn->trans_x_pdf = NormPDF(0, 2.5);
    pen_fn->trans_y_pdf = NormPDF(0, 2.5);
    pen_fn->trans_z_pdf = NormPDF(0, 10);
#else
    auto pen_fn = std::make_shared<RotMagAndTransMagPen>();
    pen_fn->rot_pdfs_per_obj.assign(1, FoldNormPDF(2.5 * kDEG2RAD, 2.5 * kDEG2RAD));
    pen_fn->trans_pdfs_per_obj.assign(1, FoldNormPDF(10, 10));
#endif

    p.lvl_1_pelvis_reg_fn = pen_fn;
    p.lvl_1_pelvis_reg_img_coeff = 0.9;
  }
  else
  {
    p.lvl_1_pelvis_bounds = { 5 * kDEG2RAD, 5 * kDEG2RAD, 5 * kDEG2RAD, 5, 5, 50 };
  }
  
  p.lvl_2_ds_factor = 0.25;
  
  p.lvl_2_smooth_kernel_size = 11;
  
  {
    const double ds_mult = p.lvl_1_ds_factor * (1.0 / p.lvl_2_ds_factor / 2.0);

    p.lvl_2_pelvis_bounds = { ds_mult * 5 * kDEG2RAD, ds_mult * 5 * kDEG2RAD, ds_mult * 5 * kDEG2RAD,
                              ds_mult * 5, ds_mult * 5, ds_mult * 50 };
  }

  p.lvl_2_pelvis_tol_fn = 0.0001;
  p.lvl_2_pelvis_tol_x = 0.0001;

  return p;
}

jhmr::tbme::FragRegiParams
jhmr::tbme::GetFragRegiParamsUsingCmdLineArgs(const bool use_old_sim_params,
                                              const bool use_known_rel_views,
                                              const bool do_est_pu_il_regi,
                                              const bool use_patch_sim,
                                              const bool use_reg)
{
  // cannot use unknown relative views with old sim params
  jhmrASSERT(!(use_old_sim_params && !use_known_rel_views));
  
  FragRegiParams p;

  if (use_old_sim_params)
  {
    jhmrASSERT(!use_reg);
    p = MakeFragRegiParamsOldSim(use_patch_sim);
  }
  else if (use_known_rel_views && !do_est_pu_il_regi)
  {
    if (use_reg)
    {
      p = MakeFragRegiParamsNewSimReg(use_patch_sim);
    }
    else
    {
      p = MakeFragRegiParamsNewSim(use_patch_sim);
    }
  }
  else
  {
    if (use_reg)
    {
      p = MakeFragRegiParamsNewSimPelvisFidReg(use_patch_sim);
    }
    else
    {
      p = MakeFragRegiParamsNewSimPelvisFid(use_patch_sim);
    }
  }

  return p;
}

std::string jhmr::tbme::GetFullProjPathH5(const size_type cut_idx, const size_type repo_idx,
                                          const std::string data_name)
{
  return fmt::sprintf("rand-plans/%02lu/repos/%02lu/projs/%s", cut_idx, repo_idx, data_name);
}

std::string jhmr::tbme::GetFullRegiPathH5(const size_type cut_idx, const size_type repo_idx,
                                          const std::string data_name,
                                          const std::string frag_regi_str,
                                          const size_type init_idx, const bool use_known_rel_views)
{
  return fmt::sprintf("%s/frag-regis/%s/%s/init-%02lu",
                      GetFullProjPathH5(cut_idx, repo_idx, data_name),
                      use_known_rel_views ? "known-rel-views" : "unk-rel-views",
                      frag_regi_str, init_idx);
} 

std::string jhmr::tbme::GetFullPelvisRegiPathH5(const size_type cut_idx, const size_type repo_idx, const std::string data_name,
                                                const std::string frag_regi_str,
                                                const size_type init_idx, const bool use_known_rel_views,
                                                const int rel_view_idx)
{
  const std::string prefix = GetFullRegiPathH5(cut_idx, repo_idx, data_name, frag_regi_str,
                                               init_idx, use_known_rel_views);

  if (use_known_rel_views || (rel_view_idx < 0))
  {
    // if rel_view_idx is not specified when not using known relative views, then we
    // are probably getting this path to get the estimated cuts segmentation
    return fmt::sprintf("%s/full-pelvis", prefix);
  }
  else
  {
    return fmt::sprintf("%s/pelvis-regi-view-%d", prefix, rel_view_idx);
  }
}

std::string jhmr::tbme::GetFullRegiDebugPathH5(const size_type cut_idx, const size_type repo_idx,
                                               const std::string data_name,
                                               const std::string frag_regi_str,
                                               const size_type init_idx, const bool use_known_rel_views)
{
  return fmt::sprintf("%s/debug",
              GetFullRegiPathH5(cut_idx, repo_idx, data_name, frag_regi_str, init_idx, use_known_rel_views));
}

std::string jhmr::tbme::GetFullPelvisRegiDebugPathH5(const size_type cut_idx, const size_type repo_idx, const std::string data_name,
                                                     const std::string frag_regi_str,
                                                     const size_type init_idx, const bool use_known_rel_views,
                                                     const int rel_view_idx)
{
  return fmt::sprintf("%s/debug",
              GetFullPelvisRegiPathH5(cut_idx, repo_idx, data_name, frag_regi_str, init_idx,
                                      use_known_rel_views, rel_view_idx));
}

void jhmr::tbme::SetupMultiObjMultiLevelFragRegi(MultiLevelMultiObjRegi* ml_mo_regi, GPUPrefsXML* gpu_prefs,
                                                 const FragRegiParams& params, std::ostream& vout)
{
  using UseOtherVolCurEstForInit = MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst;
  
  using CMAESRegi = Regi2D3DCMAES<MultiLevelMultiObjRegi::RayCaster,
                                  MultiLevelMultiObjRegi::SimMetric>;

  using BOBYQARegi = Regi2D3DBOBYQA<MultiLevelMultiObjRegi::RayCaster,
                                    MultiLevelMultiObjRegi::SimMetric>;

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();

  // TODO: this should really be taken from the fixed image indices to use field...
  const size_type num_views = ml_mo_regi->fixed_cam_imgs.size();

  // Level 1
  {
    vout << "setting up fragment regi level 1..." << std::endl;

    auto& lvl = ml_mo_regi->levels[0];
    
    lvl.ds_factor   = params.lvl_1_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics.resize(num_views);
    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      auto sm = params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 0);

      lvl.sim_metrics[view_idx] = sm;
    }
    
    // one registration for each object
    lvl.regis.resize(3);

    // pelvis
    {
      auto& regi = lvl.regis[0];
     
      regi.mov_vols    = { 0 };  // pelvis vol pose is optimized over
      regi.ref_frames  = { 0 };  // use APP ref frame
      regi.static_vols = { };    // No other object poses have been estimated yet
    
      // use pelvis initial guess for this pelvis regi
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto cmaes_regi = std::make_shared<CMAESRegi>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(params.lvl_1_pelvis_tol_x);
      cmaes_regi->set_opt_obj_fn_tol(params.lvl_1_pelvis_tol_fn);
      cmaes_regi->set_pop_size(params.lvl_1_pelvis_pop_size);
      cmaes_regi->set_sigma(params.lvl_1_pelvis_sigma);
      
      if (params.lvl_1_pelvis_reg_fn)
      {
        jhmrASSERT((params.lvl_1_pelvis_reg_img_coeff > -1.0e-8) &&
                   (params.lvl_1_pelvis_reg_img_coeff < (1.0 + 1.0e-8)));

        cmaes_regi->set_penalty_fn(params.lvl_1_pelvis_reg_fn);
        cmaes_regi->set_img_sim_penalty_coefs(params.lvl_1_pelvis_reg_img_coeff,
                                              1.0 - params.lvl_1_pelvis_reg_img_coeff);
      }
      else
      {
        cmaes_regi->set_bounds(params.lvl_1_pelvis_bounds);
      }
      
      if (params.get_lvl_1_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_1_pelvis_before_regi_callback(cmaes_regi.get()));
      }

      regi.regi = cmaes_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-0-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-0-pelvis");
    }

    // femur
    if (true)
    {
      auto& regi = lvl.regis[1];
      
      regi.mov_vols    = { 1 };  // femur vol pose is optimized over
      regi.ref_frames  = { 0 };  // use APP ref frame
      regi.static_vols = { 0 };  // Keep the pelvis pose fixed and DRR in the background

      // use current pelvis estimate for femur initial guess
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      regi.static_vol_poses = { init_guess_fn };

      auto cmaes_regi = std::make_shared<CMAESRegi>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(params.lvl_1_femur_tol_x);
      cmaes_regi->set_opt_obj_fn_tol(params.lvl_1_femur_tol_fn);
      cmaes_regi->set_pop_size(params.lvl_1_femur_pop_size);
      cmaes_regi->set_sigma(params.lvl_1_femur_sigma);
      
      if (params.lvl_1_femur_reg_fn)
      {
        jhmrASSERT((params.lvl_1_femur_reg_img_coeff > -1.0e-8) &&
                   (params.lvl_1_femur_reg_img_coeff < (1.0 + 1.0e-8)));

        cmaes_regi->set_penalty_fn(params.lvl_1_femur_reg_fn);
        cmaes_regi->set_img_sim_penalty_coefs(params.lvl_1_femur_reg_img_coeff,
                                              1.0 - params.lvl_1_femur_reg_img_coeff);
      }
      else
      {
        cmaes_regi->set_bounds(params.lvl_1_femur_bounds);
      }
      
      if (params.get_lvl_1_femur_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_1_femur_before_regi_callback(cmaes_regi.get()));
      }

      regi.regi = cmaes_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-0-femur");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-0-femur");
    }
    
    // frag
    if (true)
    {
      auto& regi = lvl.regis[2];
      
      regi.mov_vols    = { 2 };  // frag vol pose is optimized over
      regi.ref_frames  = { 0 };  // use APP ref frame
      regi.static_vols = { 0, 1 };  // keep pelvis and femur poses fixed and the DRRs in the background

      // use current pelvis estimate for fragment initial guess
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };
     
      auto femur_static_pose_fn = std::make_shared<UseOtherVolCurEstForInit>();
      femur_static_pose_fn->vol_idx = 1;

      regi.static_vol_poses = { init_guess_fn, femur_static_pose_fn };

      auto cmaes_regi = std::make_shared<CMAESRegi>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(params.lvl_1_frag_tol_x);
      cmaes_regi->set_opt_obj_fn_tol(params.lvl_1_frag_tol_fn);
      cmaes_regi->set_pop_size(params.lvl_1_frag_pop_size);
      cmaes_regi->set_sigma(params.lvl_1_frag_sigma);
      
      if (params.lvl_1_frag_reg_fn)
      {
        jhmrASSERT((params.lvl_1_frag_reg_img_coeff > -1.0e-8) &&
                   (params.lvl_1_frag_reg_img_coeff < (1.0 + 1.0e-8)));

        cmaes_regi->set_penalty_fn(params.lvl_1_frag_reg_fn);
        cmaes_regi->set_img_sim_penalty_coefs(params.lvl_1_frag_reg_img_coeff,
                                              1.0 - params.lvl_1_frag_reg_img_coeff);
      }
      else
      {
        cmaes_regi->set_bounds(params.lvl_1_frag_bounds);
      }

      if (params.get_lvl_1_frag_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_1_frag_before_regi_callback(cmaes_regi.get()));
      }

      regi.regi = cmaes_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-0-frag");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-0-frag");
    }
  }  // Level 1
            
  // Level 2
  {
    vout << "setting up level 2..." << std::endl;
    
    auto& lvl = ml_mo_regi->levels[1];
    
    lvl.ds_factor   = params.lvl_2_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics.resize(num_views);
    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      auto sm = params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 1);

      lvl.sim_metrics[view_idx] = sm;
    }
    
    // one registration for each object + one registration with all objects simultaneous
    lvl.regis.resize(4);

    // pelvis
    {
      auto& regi = lvl.regis[0];
      
      regi.mov_vols    = { 0 };     // optimize over the pelvis pose
      regi.ref_frames  = { 0 };     // optimize in the APP frame
      regi.static_vols = { 1, 2 };  // Keep the femur and fragment poses fixed, with DRRs in the background
    
      // use previous pose estimate of pelvis for the initialization 
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto femur_static_pose_fn = std::make_shared<UseOtherVolCurEstForInit>();
      femur_static_pose_fn->vol_idx = 1;
      auto frag_static_pose_fn = std::make_shared<UseOtherVolCurEstForInit>();
      frag_static_pose_fn->vol_idx = 2;

      regi.static_vol_poses = { femur_static_pose_fn, frag_static_pose_fn };

      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(params.lvl_2_pelvis_tol_x);
      bobyqa_regi->set_opt_obj_fn_tol(params.lvl_2_pelvis_tol_fn);
      bobyqa_regi->set_bounds(params.lvl_2_pelvis_bounds);
      
      if (params.get_lvl_2_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_2_pelvis_before_regi_callback(bobyqa_regi.get()));
      }

      regi.regi = bobyqa_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-1-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-1-pelvis");
    }

    // femur
    {
      auto& regi = lvl.regis[1];
      
      regi.mov_vols    = { 1 };     // optimize over the femur pose
      regi.ref_frames  = { 0 };     // optimize in the APP frame
      regi.static_vols = { 0, 2 };  // Keep the pelvis and fragment poses fixed and DRRs in the background

      // use previous pose estimate of femur for the initialization 
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 1;
      regi.init_mov_vol_poses = { init_guess_fn };
      
      auto pelvis_static_pose_fn = std::make_shared<UseOtherVolCurEstForInit>();
      pelvis_static_pose_fn->vol_idx = 0;
      auto frag_static_pose_fn = std::make_shared<UseOtherVolCurEstForInit>();
      frag_static_pose_fn->vol_idx = 2;

      regi.static_vol_poses = { pelvis_static_pose_fn, frag_static_pose_fn };
      
      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(params.lvl_2_femur_tol_x);
      bobyqa_regi->set_opt_obj_fn_tol(params.lvl_2_femur_tol_fn);
      bobyqa_regi->set_bounds(params.lvl_2_femur_bounds);

      regi.regi = bobyqa_regi;
      
      if (params.get_lvl_2_femur_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_2_femur_before_regi_callback(bobyqa_regi.get()));
      }
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-1-femur");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-1-femur");
    }
    
    // frag
    {
      auto& regi = lvl.regis[2];
      
      regi.mov_vols    = { 2 };     // optimize over the fragment pose
      regi.ref_frames  = { 0 };     // optimize in the APP frame
      regi.static_vols = { 0, 1 };  // Keep pelvis and femur poses fixed and DRRs in the background

      // use previous pose estimate of fragment for the initialization 
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 2;
      regi.init_mov_vol_poses = { init_guess_fn };
      
      auto pelvis_static_pose_fn = std::make_shared<UseOtherVolCurEstForInit>();
      pelvis_static_pose_fn->vol_idx = 0;
      auto femur_static_pose_fn = std::make_shared<UseOtherVolCurEstForInit>();
      femur_static_pose_fn->vol_idx = 1;

      regi.static_vol_poses = { pelvis_static_pose_fn, femur_static_pose_fn };
      
      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(params.lvl_2_frag_tol_x);
      bobyqa_regi->set_opt_obj_fn_tol(params.lvl_2_frag_tol_fn);
      bobyqa_regi->set_bounds(params.lvl_2_frag_bounds);
      
      if (params.get_lvl_2_frag_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_2_frag_before_regi_callback(bobyqa_regi.get()));
      }

      regi.regi = bobyqa_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-1-frag");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-1-frag");
    }
    
    // all objects
    {
      auto& regi = lvl.regis[3];
      
      regi.mov_vols    = { 0, 1, 2 };  // optimize over pelvis, femur, frag poses simultaneously
      regi.ref_frames  = { 0, 0, 0 };  // optimize in the APP frame
      regi.static_vols = { };          // no objects are kept fixed
    
      // use previous pose estimates for the shapes initialization 
      auto pelvis_init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      pelvis_init_guess_fn->vol_idx = 0;
      auto femur_init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      femur_init_guess_fn->vol_idx = 1;
      auto frag_init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      frag_init_guess_fn->vol_idx = 2;

      regi.init_mov_vol_poses = { pelvis_init_guess_fn, femur_init_guess_fn, frag_init_guess_fn };

      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(params.lvl_2_all_tol_x);
      bobyqa_regi->set_opt_obj_fn_tol(params.lvl_2_all_tol_fn);
      bobyqa_regi->set_bounds(params.lvl_2_all_bounds);
      
      if (params.get_lvl_2_all_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_2_all_before_regi_callback(bobyqa_regi.get()));
      }

      regi.regi = bobyqa_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-1-all");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-1-all");
    }
  }  // Level 2
}

void jhmr::tbme::SetupPelvisMultiLevelRegi(MultiLevelMultiObjRegi* ml_mo_regi, GPUPrefsXML* gpu_prefs,
                                           const FragRegiParams& params, std::ostream& vout)
{
  using UseOtherVolCurEstForInit = MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst;
  
  using CMAESRegi = Regi2D3DCMAES<MultiLevelMultiObjRegi::RayCaster,
                                  MultiLevelMultiObjRegi::SimMetric>;

  using BOBYQARegi = Regi2D3DBOBYQA<MultiLevelMultiObjRegi::RayCaster,
                                    MultiLevelMultiObjRegi::SimMetric>;

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();

  const size_type num_views = ml_mo_regi->fixed_cam_imgs.size();

  // Level 1
  {
    vout << "setting up pelvis regi level 1..." << std::endl;

    auto& lvl = ml_mo_regi->levels[0];
    
    lvl.ds_factor   = params.lvl_1_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics.resize(num_views);
    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      auto sm = params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 0);
      lvl.sim_metrics[view_idx] = sm;
    }
    
    // one registration for full pelvis 
    lvl.regis.resize(1);

    // pelvis
    {
      auto& regi = lvl.regis[0];
     
      regi.mov_vols    = { 0 };  // pelvis vol pose is optimized over
      regi.ref_frames  = { 0 };  // use APP ref frame
      regi.static_vols = { };    // No other objects exist
    
      // use pelvis initial guess for this pelvis regi
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto cmaes_regi = std::make_shared<CMAESRegi>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(params.lvl_1_pelvis_tol_x);
      cmaes_regi->set_opt_obj_fn_tol(params.lvl_1_pelvis_tol_fn);
      cmaes_regi->set_pop_size(params.lvl_1_pelvis_pop_size);
      cmaes_regi->set_sigma(params.lvl_1_pelvis_sigma);
      
      if (params.lvl_1_pelvis_reg_fn)
      {
        jhmrASSERT((params.lvl_1_pelvis_reg_img_coeff > -1.0e-8) &&
                   (params.lvl_1_pelvis_reg_img_coeff < (1.0 + 1.0e-8)));

        cmaes_regi->set_penalty_fn(params.lvl_1_pelvis_reg_fn);
        cmaes_regi->set_img_sim_penalty_coefs(params.lvl_1_pelvis_reg_img_coeff,
                                              1.0 - params.lvl_1_pelvis_reg_img_coeff);
      }
      else
      {
        cmaes_regi->set_bounds(params.lvl_1_pelvis_bounds);
      }
      
      if (params.get_lvl_1_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_1_pelvis_before_regi_callback(cmaes_regi.get()));
      }

      regi.regi = cmaes_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-0-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-0-pelvis");
    }
  }  // Level 1
            
  // Level 2
  {
    vout << "setting up pelvis regi level 2..." << std::endl;
    
    auto& lvl = ml_mo_regi->levels[1];
    
    lvl.ds_factor   = params.lvl_2_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics.resize(num_views);
    for (size_type view_idx = 0; view_idx < num_views; ++view_idx)
    {
      auto sm = params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 1);
      lvl.sim_metrics[view_idx] = sm;
    }
    
    // one registration for the full pelvis
    lvl.regis.resize(1);

    // pelvis
    {
      auto& regi = lvl.regis[0];
      
      regi.mov_vols    = { 0 };     // optimize over the pelvis pose
      regi.ref_frames  = { 0 };     // optimize in the APP frame
      regi.static_vols = { };    // No other objects exist
    
      // use previous pose estimate of pelvis for the initialization 
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(params.lvl_2_pelvis_tol_x);
      bobyqa_regi->set_opt_obj_fn_tol(params.lvl_2_pelvis_tol_fn);
      bobyqa_regi->set_bounds(params.lvl_2_pelvis_bounds);
      
      if (params.get_lvl_2_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_2_pelvis_before_regi_callback(bobyqa_regi.get()));
      }

      regi.regi = bobyqa_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-1-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-1-pelvis");
    }
  }  // Level 2
}

void jhmr::tbme::SetupPelvisMultiLevelRegiSingleAPView(MultiLevelMultiObjRegi* ml_mo_regi,
                                                       GPUPrefsXML* gpu_prefs,
                                                       const FragRegiParams& params, std::ostream& vout)
{
  using UseOtherVolCurEstForInit = MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst;
  
  using CMAESRegi = Regi2D3DCMAES<MultiLevelMultiObjRegi::RayCaster,
                                  MultiLevelMultiObjRegi::SimMetric>;

  using BOBYQARegi = Regi2D3DBOBYQA<MultiLevelMultiObjRegi::RayCaster,
                                    MultiLevelMultiObjRegi::SimMetric>;

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();

  // Level 1
  {
    vout << "setting up pelvis regi level 1..." << std::endl;

    auto& lvl = ml_mo_regi->levels[0];
    
    lvl.ds_factor   = params.lvl_1_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics = { params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 0) };
    
    // one registration for full pelvis 
    lvl.regis.resize(1);

    // pelvis
    {
      auto& regi = lvl.regis[0];
     
      regi.mov_vols    = { 0 };  // pelvis vol pose is optimized over
      regi.ref_frames  = { 0 };  // use ref frame that is camera aligned
      regi.static_vols = { };    // No other objects exist
    
      // use pelvis initial guess for this pelvis regi
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto cmaes_regi = std::make_shared<CMAESRegi>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(params.lvl_1_pelvis_tol_x);
      cmaes_regi->set_opt_obj_fn_tol(params.lvl_1_pelvis_tol_fn);
      cmaes_regi->set_pop_size(params.lvl_1_pelvis_pop_size);
      cmaes_regi->set_sigma(params.lvl_1_pelvis_sigma);
      
      if (params.lvl_1_pelvis_reg_fn)
      {
        jhmrASSERT((params.lvl_1_pelvis_reg_img_coeff > -1.0e-8) &&
                   (params.lvl_1_pelvis_reg_img_coeff < (1.0 + 1.0e-8)));

        cmaes_regi->set_penalty_fn(params.lvl_1_pelvis_reg_fn);
        cmaes_regi->set_img_sim_penalty_coefs(params.lvl_1_pelvis_reg_img_coeff,
                                              1.0 - params.lvl_1_pelvis_reg_img_coeff);
      }
      else
      {
        cmaes_regi->set_bounds(params.lvl_1_pelvis_bounds);
      }
      
      if (params.get_lvl_1_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_1_pelvis_before_regi_callback(cmaes_regi.get()));
      }

      regi.regi = cmaes_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-0-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-0-pelvis");
    }
  }  // Level 1
            
  // Level 2
  {
    vout << "setting up pelvis regi level 2..." << std::endl;
    
    auto& lvl = ml_mo_regi->levels[1];
    
    lvl.ds_factor   = params.lvl_2_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics = { params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 1) };
    
    // one registration for the full pelvis
    lvl.regis.resize(1);

    // pelvis
    {
      auto& regi = lvl.regis[0];
      
      regi.mov_vols    = { 0 };     // optimize over the pelvis pose
      regi.ref_frames  = { 0 };     // optimize in frame that is camera aligned
      regi.static_vols = { };    // No other objects exist
    
      // use previous pose estimate of pelvis for the initialization 
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(params.lvl_2_pelvis_tol_x);
      bobyqa_regi->set_opt_obj_fn_tol(params.lvl_2_pelvis_tol_fn);
      bobyqa_regi->set_bounds(params.lvl_2_pelvis_bounds);
      
      if (params.get_lvl_2_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_2_pelvis_before_regi_callback(bobyqa_regi.get()));
      }

      regi.regi = bobyqa_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-1-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-1-pelvis");
    }
  }  // Level 2
}

void jhmr::tbme::SetupPelvisMultiLevelRegiSingleRotFromAPView(MultiLevelMultiObjRegi* ml_mo_regi,
                                                              GPUPrefsXML* gpu_prefs,
                                                              const FragRegiParams& params,
                                                              const FrameTransform& prev_view_pelvis_pose,
                                                              std::ostream& vout)
{
  using UseOtherVolCurEstForInit = MultiLevelMultiObjRegi::Level::SingleRegi::InitPosePrevPoseEst;
  
  using CMAESRegi = Regi2D3DCMAES<MultiLevelMultiObjRegi::RayCaster,
                                  MultiLevelMultiObjRegi::SimMetric>;

  using BOBYQARegi = Regi2D3DBOBYQA<MultiLevelMultiObjRegi::RayCaster,
                                    MultiLevelMultiObjRegi::SimMetric>;

  using ExhaustiveRegi = Regi2D3DExhaustive<MultiLevelMultiObjRegi::RayCaster,
                                            MultiLevelMultiObjRegi::SimMetric>;

  auto se3_vars = std::make_shared<SE3OptVarsLieAlg<double>>();

  // Level 1
  {
    vout << "setting up pelvis regi level 1..." << std::endl;

    auto& lvl = ml_mo_regi->levels[0];
    
    lvl.ds_factor   = params.lvl_1_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics = { params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 0) };
    
    // one registration for full pelvis 
    lvl.regis.resize(2);
    
    // brute-force C-Arm orbit for pelvis
    {
      auto& regi = lvl.regis[0];
     
      regi.mov_vols    = { 0 };  // pelvis vol pose is optimized over
      regi.ref_frames  = { 1 };  // orbit ref frame
      regi.static_vols = { };    // No other objects exist
    
      // use pelvis estimate from previous regi
      // I believe this is not actually used in the search, but will be useful for making
      // a debug movie.
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto ex_regi = std::make_shared<ExhaustiveRegi>();
      ex_regi->set_opt_vars(se3_vars);

      constexpr size_type kNUM_ROTS = 181;

      constexpr double kROT_INC   = 1.0 * kDEG2RAD;
      constexpr double kROT_START = -90 * kDEG2RAD;

      FrameTransformList rot_xforms(kNUM_ROTS);
     
      double rot_ang_rad = kROT_START;
      for (size_type rot_idx = 0; rot_idx < kNUM_ROTS; ++rot_idx, rot_ang_rad += kROT_INC)
      {
        rot_xforms[rot_idx] = EulerRotXFrame<CoordScalar>(rot_ang_rad);
      }

      ex_regi->set_cam_wrt_vols(ExhaustiveRegi::ListOfFrameTransformLists(1, rot_xforms));

      regi.regi = ex_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-0-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-0-pelvis");
    }

    // classic pelvis
    {
      auto& regi = lvl.regis[1];
     
      regi.mov_vols    = { 0 };  // pelvis vol pose is optimized over
      regi.ref_frames  = { 0 };  // use ref frame that is camera aligned
      regi.static_vols = { };    // No other objects exist
    
      // use pelvis estimate from previous search to initialize this pelvis regi
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto cmaes_regi = std::make_shared<CMAESRegi>();
      cmaes_regi->set_opt_vars(se3_vars);
      cmaes_regi->set_opt_x_tol(params.lvl_1_pelvis_tol_x);
      cmaes_regi->set_opt_obj_fn_tol(params.lvl_1_pelvis_tol_fn);
      cmaes_regi->set_pop_size(params.lvl_1_pelvis_pop_size);
      cmaes_regi->set_sigma(params.lvl_1_pelvis_sigma);
      
      if (params.lvl_1_pelvis_reg_fn)
      {
        jhmrASSERT((params.lvl_1_pelvis_reg_img_coeff > -1.0e-8) &&
                   (params.lvl_1_pelvis_reg_img_coeff < (1.0 + 1.0e-8)));

        cmaes_regi->set_penalty_fn(params.lvl_1_pelvis_reg_fn);
        cmaes_regi->set_img_sim_penalty_coefs(params.lvl_1_pelvis_reg_img_coeff,
                                              1.0 - params.lvl_1_pelvis_reg_img_coeff);
      }
      else
      {
        cmaes_regi->set_bounds(params.lvl_1_pelvis_bounds);
      }
      
      if (params.get_lvl_1_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_1_pelvis_before_regi_callback(cmaes_regi.get()));
      }

      regi.regi = cmaes_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-0-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-0-pelvis");
    }
  }  // Level 1
            
  // Level 2
  {
    vout << "setting up pelvis regi level 2..." << std::endl;
    
    auto& lvl = ml_mo_regi->levels[1];
    
    lvl.ds_factor   = params.lvl_2_ds_factor;
    lvl.ray_caster  = std::make_shared<RayCasterLineIntGPU>(gpu_prefs->ctx, gpu_prefs->queue);
    //lvl.ray_caster  = std::make_shared<jhmr::CameraRayCasterCPULineIntegral<PixelScalar,CoordScalar>>();
    
    lvl.sim_metrics = { params.get_sim_metric_obj(gpu_prefs->ctx, gpu_prefs->queue, params, 1) };
    
    // one registration for the full pelvis
    lvl.regis.resize(1);

    // pelvis
    {
      auto& regi = lvl.regis[0];
      
      regi.mov_vols    = { 0 };     // optimize over the pelvis pose
      regi.ref_frames  = { 0 };     // optimize in frame that is camera aligned
      regi.static_vols = { };    // No other objects exist
    
      // use previous pose estimate of pelvis for the initialization 
      auto init_guess_fn = std::make_shared<UseOtherVolCurEstForInit>();
      init_guess_fn->vol_idx = 0;
      regi.init_mov_vol_poses = { init_guess_fn };

      auto bobyqa_regi = std::make_shared<BOBYQARegi>();
      bobyqa_regi->set_opt_vars(se3_vars);
      bobyqa_regi->set_opt_x_tol(params.lvl_2_pelvis_tol_x);
      bobyqa_regi->set_opt_obj_fn_tol(params.lvl_2_pelvis_tol_fn);
      bobyqa_regi->set_bounds(params.lvl_2_pelvis_bounds);
      
      if (params.get_lvl_2_pelvis_before_regi_callback)
      {
        regi.fns_to_call_right_before_regi_run.assign(1,
            params.get_lvl_2_pelvis_before_regi_callback(bobyqa_regi.get()));
      }

      regi.regi = bobyqa_regi;
      
      //regi.regi->set_debug_write_fixed_img_edge_overlays(true);
      //MakeDirRecursive("regi-debug-lvl-1-pelvis");
      //regi.regi->set_debug_output_dir_path("regi-debug-lvl-1-pelvis");
    }
  }  // Level 2
}

cv::Mat jhmr::tbme::OverlayLabelMapAsRGBOnGrayscale(const cv::Mat& gray_img,
                                                    const itk::Image<LabelScalar,2>* labels,
                                                    const double mask_alpha)
{
  jhmrASSERT((mask_alpha >= 0) && (mask_alpha <= 1));

  const double img_alpha = 1.0 - mask_alpha;

  const auto seg_lut = GenericAnatomyLUT();
            
  cv::Mat img_bgr;
  cv::cvtColor(gray_img, img_bgr, cv::COLOR_GRAY2BGR);

  auto mask_remap = RemapITKLabelMap<unsigned char, 2, itk::RGBPixel<unsigned char>>(
                                labels, seg_lut);

  cv::Mat mask_remap_ocv(gray_img.rows, gray_img.cols, CV_8UC3);
 
  const unsigned char* src_mask_buf = labels->GetBufferPointer();

  const itk::RGBPixel<unsigned char>* src_mask_remap_buf = mask_remap->GetBufferPointer();

  for (int r = 0; r < gray_img.rows;
       ++r, src_mask_buf += gray_img.cols, src_mask_remap_buf += gray_img.cols)
  {
    unsigned char* dst_row = &img_bgr.at<unsigned char>(r,0);

    for (int c = 0; c < gray_img.cols; ++c)
    {
      if (src_mask_buf[c])
      {
        const auto& src_p = src_mask_remap_buf[c];

        const size_type off = c * 3;

        dst_row[off] = static_cast<unsigned char>(std::round((img_alpha * dst_row[off]) +
                                                             (mask_alpha * src_p[2])));
        dst_row[off + 1] = static_cast<unsigned char>(std::round((img_alpha * dst_row[off + 1]) +
                                                                 (mask_alpha * src_p[1])));
        dst_row[off + 2] = static_cast<unsigned char>(std::round((img_alpha * dst_row[off + 2]) +
                                                                 (mask_alpha * src_p[0])));
      }
    }
  }

  return img_bgr;
}

namespace
{

using namespace jhmr;
using namespace jhmr::tbme;

Countdown::CallbackFn MakeTimeLimitCallback(H5::H5File& h5, std::ostream& vout)
{
  std::ostream* tmp_out = &std::cerr;

  return [&h5, tmp_out] ()
  {
    *tmp_out << "TIME LIMIT EXCEEDED!" << std::endl;

    h5.flush(H5F_SCOPE_GLOBAL);
    h5.close();

    std::exit(0);
  };
}

}  // un-named

void jhmr::tbme::UpdateCountdownH5(Countdown* time_limit, H5::H5File& h5, std::ostream& vout)
{
  time_limit->set_limit_exceeded_fn(MakeTimeLimitCallback(h5, vout));
}

std::unique_ptr<jhmr::Countdown>
jhmr::tbme::MakeCountdown(const std::string& time_format,
                          H5::H5File& h5,
                          std::ostream& vout)
{
  std::unique_ptr<Countdown> time_limit;
  if (!time_format.empty())
  {
    vout << "setting a runtime limit! " << time_format << std::endl;
    
    time_limit.reset(new Countdown(time_format, MakeTimeLimitCallback(h5, vout)));
    
    vout << "seconds remaining: " << time_limit->seconds_remaining() << std::endl;
  }
  else
  {
    vout << "no runtime limit set..." << std::endl;
  }
  
  return std::move(time_limit);
}

jhmr::tbme::EstCutsAndCreateFragHelper::EstCutsAndCreateFragHelper(const std::string& est_cuts_exe_path,
                                                                   const std::string& create_frag_seg_exe_path,
                                                                   const std::string& work_dir)
  : est_cuts_exe_path_(est_cuts_exe_path),
    create_frag_seg_exe_path_(create_frag_seg_exe_path),

    work_dir_(work_dir),

    tmp_full_pelvis_seg_path_  (fmt::sprintf("%s/tmp_full_pelvis_seg.nii.gz", work_dir)),
    tmp_hemi_pelvis_seg_path_  (fmt::sprintf("%s/tmp_hemi_pelvis_seg.nii.gz", work_dir)),
    tmp_final_cuts_seg_path_   (fmt::sprintf("%s/tmp_cuts_seg.nii.gz",        work_dir)),
    tmp_final_cuts_xml_path_   (fmt::sprintf("%s/tmp_final_cuts.xml",         work_dir)),
    tmp_init_cuts_xml_path_    (fmt::sprintf("%s/tmp_init_cuts.xml",          work_dir)),
    tmp_lands_path_            (fmt::sprintf("%s/tmp_lands.fcsv",             work_dir)),
    tmp_regi_pose_path_        (fmt::sprintf("%s/tmp_regi_pose.h5",           work_dir)),
    tmp_label_projs_path_      (fmt::sprintf("%s/tmp_label_projs.h5",         work_dir)),
    tmp_invalid_cuts_path_     (fmt::sprintf("%s/tmp_invalid_cuts.nii.gz",    work_dir)),
    tmp_il_recon_pts_fcsv_path_(fmt::sprintf("%s/cut_pts_ilium.fcsv",         work_dir)),
    tmp_pu_recon_pts_fcsv_path_(fmt::sprintf("%s/cut_pts_pubis.fcsv",         work_dir)),
    tmp_log_file_path_         (fmt::sprintf("%s/tmp_log.txt",                work_dir))
{

}

std::string jhmr::tbme::EstCutsAndCreateFragHelper::verbose_arg_str() const
{
  return "-v";
  //return "";
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_full_pelvis_seg(const LabelVol* seg)
{
  WriteITKImageToDisk(seg, tmp_full_pelvis_seg_path_);
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_hemipelvis_scalar_seg(const Vol* hemi)
{
  WriteITKImageToDisk(hemi, tmp_hemi_pelvis_seg_path_);
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_invalid_frag_mask(const LabelVol* mask)
{
  WriteITKImageToDisk(mask, tmp_invalid_cuts_path_);
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_anat_lands(const LandMap3& lands)
{
  WriteFCSVFileFromNamePtMap(tmp_lands_path_, lands.begin(), lands.end());
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_regi(const FrameTransform& regi_xform)
{
  WriteITKAffineTransform(tmp_regi_pose_path_, regi_xform);
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_orig_cuts(const PAOCuts& cuts_def, const PAODispInfo& cuts_disp)
{
  WritePAOCutPlanesXML(cuts_def, tmp_init_cuts_xml_path_, &cuts_disp);
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_est_cuts(const PAOCuts& cuts_def, const PAODispInfo& cuts_disp)
{
  WritePAOCutPlanesXML(cuts_def, tmp_final_cuts_xml_path_, &cuts_disp);
}

void jhmr::tbme::EstCutsAndCreateFragHelper::set_proj_labels(const std::vector<CutLabelProjData>& proj_labels)
{
  WriteProjData(tmp_label_projs_path_, proj_labels);
}

std::tuple<jhmr::LabelVolPtr,double>
jhmr::tbme::EstCutsAndCreateFragHelper::create_frag_seg() const
{
  jhmrASSERT((side_str == "left") || (side_str == "right"));

  jhmrASSERT(Path(tmp_full_pelvis_seg_path_).exists());
  jhmrASSERT(Path(tmp_lands_path_).exists());
  jhmrASSERT(Path(tmp_final_cuts_xml_path_).exists());
  jhmrASSERT(Path(tmp_invalid_cuts_path_).exists());

  const std::string cmd_str = fmt::sprintf("%s %s %s %s %s - %s --invalid-mask %s %s --time > %s",
                                           create_frag_seg_exe_path_,
                                           tmp_full_pelvis_seg_path_, tmp_lands_path_, tmp_final_cuts_xml_path_,
                                           side_str, tmp_final_cuts_seg_path_, tmp_invalid_cuts_path_,
                                           verbose_arg_str(),
                                           tmp_log_file_path_);

  dout() << "invoking create frag exe..." << std::endl;
  if (print_cmds)
  {
    dout() << "    " << cmd_str << std::endl;
  }

  if (std::system(cmd_str.c_str()))
  {
    jhmrThrow("fragment seg creation exe failed!");
  }

  const double label_map_time_secs = get_time_from_log();

  dout() << "reading tmp cuts seg..." << std::endl;
  return std::make_tuple(ReadITKImageFromDisk<LabelVol>(tmp_final_cuts_seg_path_), label_map_time_secs);
}

std::tuple<jhmr::tbme::PAOCuts,jhmr::tbme::PAODispInfo,jhmr::PtList3,jhmr::PtList3,double>
jhmr::tbme::EstCutsAndCreateFragHelper::est_cuts() const
{
  jhmrASSERT((side_str == "left") || (side_str == "right"));

  jhmrASSERT(Path(tmp_hemi_pelvis_seg_path_).exists());
  jhmrASSERT(Path(tmp_init_cuts_xml_path_).exists());
  jhmrASSERT(Path(tmp_lands_path_).exists());
  jhmrASSERT(Path(tmp_label_projs_path_).exists());
  jhmrASSERT(Path(tmp_regi_pose_path_).exists());

  // NOTE:
  //  Rays are cast from detector to source
  //  f --> entry point on bone surface of ray
  //  b --> exit point on bone surface of ray
  //  So if patient is in supine position, f/entry ray will intersect more anteriorly
  //  and the b/exit ray will intersect more posteriorly.

  const std::string il_exit_entry_arg_str = no_il_exit_entry ? "" : "--ilium-2d-fb 3";
  const std::string il_entry_arg_str      = no_il_entry      ? "" : "--ilium-2d-f 2";
  const std::string il_exit_arg_str       = no_il_exit       ? "" : "--ilium-2d-b 1";
  
  const std::string pu_exit_entry_arg_str = no_pu_exit_entry ? "" : "--pubis-2d-fb 6";
  const std::string pu_entry_arg_str      = no_pu_entry      ? "" : "--pubis-2d-f 5";
  const std::string pu_exit_arg_str       = no_pu_exit       ? "" : "--pubis-2d-b 4";

  std::string ransac_flags;
  std::string backtrack_flags;

  if (true)
  {
    //ransac_flags = "--no-ransac --no-crude-outlier";
    ransac_flags = "--ransac-il-con-size 0.85 --ransac-pu-con-size 0.6";
    backtrack_flags = "--num-backtrack-steps 10";
  }

  const std::string cmd_str = fmt::sprintf("%s "
                                           "%s %s %s "
                                           "%s %s %s "
                                           "%s %s %s %s %s %s %lu - %lu -  %s %s"
                                           " --write-fcsv --fcsv-dir %s"
                                           " %s %s --time > %s",
                                           est_cuts_exe_path_,
                                           il_exit_entry_arg_str,
                                           il_entry_arg_str,
                                           il_exit_arg_str,
                                           pu_exit_entry_arg_str,
                                           pu_entry_arg_str,
                                           pu_exit_arg_str,
                                           tmp_hemi_pelvis_seg_path_,
                                           tmp_init_cuts_xml_path_,
                                           tmp_lands_path_,
                                           side_str,
                                           tmp_label_projs_path_,
                                           tmp_regi_pose_path_,
                                           il_proj_idx,
                                           pu_proj_idx,
                                           tmp_final_cuts_xml_path_,
                                           verbose_arg_str(),
                                           work_dir_,
                                           ransac_flags,
                                           backtrack_flags,
                                           tmp_log_file_path_);

  dout() << "invoking cuts est exe..." << std::endl;
  if (print_cmds)
  {
    dout() << "    " << cmd_str << std::endl;
  }

  if (std::system(cmd_str.c_str()))
  {
    jhmrThrow("Cut fit from backproj exe failed!!");
  }
  
  PAOCuts cuts_def;
  PAODispInfo cuts_disp;

  dout() << "reading est cuts def from temp..." << std::endl;
  ReadPAOCutPlanesXML(tmp_final_cuts_xml_path_, &cuts_def, &cuts_disp);
 
  dout() << "reading the reconstructed 3D ilium cut points..." << std::endl;
  const auto il_cut_pts = ReadFCSVFilePts<Pt3>(tmp_il_recon_pts_fcsv_path_);
  
  dout() << "reading the reconstructed 3D pubis cut points..." << std::endl;
  const auto pu_cut_pts = ReadFCSVFilePts<Pt3>(tmp_pu_recon_pts_fcsv_path_);

  const double est_cuts_run_secs = get_time_from_log();

  return std::make_tuple(cuts_def, cuts_disp, il_cut_pts, pu_cut_pts, est_cuts_run_secs);
}

double jhmr::tbme::EstCutsAndCreateFragHelper::get_time_from_log() const
{
  double time_secs = 0;
    
  dout() << "parsing runtime..." << std::endl;
    
  std::ifstream in(tmp_log_file_path_);
  
  const auto log_lines = GetNonEmptyLinesFromStream(in);
  
  for (const auto& l : log_lines)
  {
    const auto toks = StringSplit(l);

    if (toks.size() > 1)
    {
      if (toks[0] == "time:")
      {
        time_secs = StringCast<double>(toks[1]);
        break;
      }
    }
  }

  return time_secs;
}

jhmr::FrameTransform
jhmr::tbme::CorrPtCloudRegi(const LandMap3& dst_lands_map,
                            const LandMap3& src_lands_map)
{
  using Pt3List = std::vector<Pt3>;

  Pt3List dst_lands;
  Pt3List src_lands;

  CreateCorrespondencePointLists(dst_lands_map, src_lands_map, &dst_lands, &src_lands);
  jhmrASSERT(!dst_lands.empty());

  FrameTransform xform = FrameTransform::Identity();
  RigidRegiQuatMeth(src_lands, dst_lands, &xform, 1.0);
  
  return xform;
}

namespace
{

using namespace jhmr;
using namespace jhmr::tbme;

LandMap3 ExtractPtsPrefixSuffix(const LandMap3& src, const std::string& prefix, const std::string& suffix)
{
  const size_type p_len = prefix.size();
  const size_type s_len = suffix.size();

  const size_type min_name_len = p_len + s_len + 1;

  LandMap3 dst;
  dst.reserve(src.size());

  for (const auto& kv : src)
  {
    const size_type name_len = kv.first.size();
    
    if (name_len >= min_name_len)
    {
      if (!p_len || (kv.first.substr(0,p_len) == prefix))
      {
        if (!s_len || (kv.first.substr(name_len - s_len) == suffix))
        {
          dst.insert(kv);
        }
      }
    }
  }

  return dst;
}

LandMap3 ExtractPtsValidInds(const LandMap3& src, const std::vector<char>& valid_inds)
{
  LandMap3 dst;
  dst.reserve(src.size());

  for (const auto& kv : src)
  {
    const size_type len = kv.first.size();

    if (len > 1)
    {
      const auto i = kv.first.find('-');
      if ((i != std::string::npos) && (i < (len - 1)) &&
          (std::find(valid_inds.begin(), valid_inds.end(), kv.first[i+1]) != valid_inds.end()))
      {
        dst.insert(kv);
      }
    }
  }

  return dst;
}

}  // un-named

jhmr::LandMap3 jhmr::tbme::ExtractAllIliumPts(const LandMap3& src)
{
  return ExtractPtsPrefixSuffix(src, "IL-", "");
}

jhmr::LandMap3 jhmr::tbme::ExtractRightIliumPts(const LandMap3& src)
{
  return ExtractPtsPrefixSuffix(src, "IL-", "-R");
}

jhmr::LandMap3 jhmr::tbme::ExtractLeftIliumPts(const LandMap3& src)
{
  return ExtractPtsPrefixSuffix(src, "IL-", "-L");
}

jhmr::LandMap3 jhmr::tbme::ExtractIliumPtsSided(const LandMap3& src, const bool is_left)
{
  return is_left ? ExtractLeftIliumPts(src) : ExtractRightIliumPts(src);
}

jhmr::LandMap3 jhmr::tbme::ExtractAllFragPts(const LandMap3& src)
{
  return ExtractPtsPrefixSuffix(src, "FR-", "");
}

jhmr::LandMap3 jhmr::tbme::ExtractRightFragPts(const LandMap3& src)
{
  return ExtractPtsPrefixSuffix(src, "FR-", "-R");
}

jhmr::LandMap3 jhmr::tbme::ExtractLeftFragPts(const LandMap3& src)
{
  return ExtractPtsPrefixSuffix(src, "FR-", "-L");
}

jhmr::LandMap3 jhmr::tbme::ExtractFragPtsSided(const LandMap3& src, const bool is_left)
{
  return is_left ? ExtractLeftFragPts(src) : ExtractRightFragPts(src);
}

jhmr::LandMap3 jhmr::tbme::ExtractPtsSided(const LandMap3& src, const bool is_left)
{
  return ExtractPtsPrefixSuffix(src, "", is_left ? "-L" : "-R");
}

std::unordered_map<std::string,Eigen::Matrix<double,3,1>> jhmr::tbme::LandMapDouble(const LandMap3& f)
{
  std::unordered_map<std::string,Eigen::Matrix<double,3,1>> d;
  d.reserve(f.size());

  for (const auto& kv : f)
  {
    d.insert(decltype(d)::value_type(kv.first, kv.second.cast<double>()));
  }

  return d;
}

std::unordered_map<std::string,Eigen::Matrix<double,2,1>> jhmr::tbme::LandMapDouble(const LandMap2& f)
{
  std::unordered_map<std::string,Eigen::Matrix<double,2,1>> d;
  d.reserve(f.size());

  for (const auto& kv : f)
  {
    d.insert(decltype(d)::value_type(kv.first, kv.second.cast<double>()));
  }

  return d;
}

jhmr::LandMap2 jhmr::tbme::LandMap3To2(const LandMap3& src)
{
  LandMap2 dst;
  dst.reserve(src.size());

  for (const auto& kv : src)
  {
    dst.insert(LandMap2::value_type(kv.first, kv.second.head(2)));
  }

  return dst;
}

jhmr::LandMap3 jhmr::tbme::ExtractSmallBBs(const LandMap3& src)
{
  const std::vector<char> small_inds = { '1', '2', '3', '4' };
  return ExtractPtsValidInds(src, small_inds);
}

jhmr::LandMap3 jhmr::tbme::ExtractLargeBBs(const LandMap3& src)
{
  const std::vector<char> large_inds = { '5', '6', '7', '8' };
  return ExtractPtsValidInds(src, large_inds);
}

std::vector<jhmr::tbme::ProjData>
jhmr::tbme::GetCamAndBBLands(const H5::Group& g, std::ostream& vout,
                             const bool include_anat_lands,
                             const bool include_pixels)
{
  const bool has_bbs_2d = ObjectInGroupH5("bb-lands", g);

  H5::Group bb_g;
  
  if (has_bbs_2d)
  {
    bb_g = g.openGroup("bb-lands");
  }
  else
  {
    vout << "Manual BB lands not present!" << std::endl;
  }

  vout << "reading cams..." << std::endl;
 
  constexpr size_type kMAX_NUM_PROJS = 100;

  std::vector<ProjData> pd;

  for (size_type i = 0; i < kMAX_NUM_PROJS; ++i)
  {
    vout << "  proj " << i << std::endl;
 
    const std::string proj_str = fmt::sprintf("proj-%lu", i+1);

    if (ObjectInGroupH5(proj_str, g))
    {
      const H5::Group proj_g = g.openGroup(proj_str);
    
      vout << "     cam..." << std::endl;

      ProjData cur_pd = include_pixels ? ReadProjDataH5<PixelScalar>(proj_g)[0]
                                       : std::get<0>(ReadProjInfoH5<PixelScalar>(proj_g))[0];

      vout << "     cios meta..." << std::endl;
      const auto meta = ReadCIOSMetaH5(proj_g.openGroup("cios-meta"));
      
      LandMap3 bb_lands_fcsv;

      if (has_bbs_2d)
      {
        vout << "     bbs..." << std::endl;
        bb_lands_fcsv = ReadLandmarksMapH5<Pt3>(bb_g.openGroup(proj_str));
    
        vout << "        converting to pixel units applying any flips required by CIOS meta..." << std::endl;
        UpdateLandmarkMapForCIOSFusion(meta, bb_lands_fcsv.begin(), bb_lands_fcsv.end());
      }

      if (include_anat_lands)
      {
        const std::string anat_lands_name = fmt::sprintf("lands-%lu", i+1);

        if (ObjectInGroupH5(anat_lands_name, g))
        {
          vout << "    reading anat landmarks..." << std::endl;
          auto anat_lands_fcsv = ReadLandmarksMapH5<Pt3>(g.openGroup(anat_lands_name));
      
          vout << "        converting to pixel units applying any flips required by CIOS meta..." << std::endl;
          UpdateLandmarkMapForCIOSFusion(meta, anat_lands_fcsv.begin(), anat_lands_fcsv.end());
        
          vout << "        merging into BB landmarks..." << std::endl;
          bb_lands_fcsv.insert(anat_lands_fcsv.begin(), anat_lands_fcsv.end());
        }
        else
        {
          vout << "    no anat landmarks for this view." << std::endl;
        }
      }

      vout << "        adding to pd..." << std::endl;

      auto& lands = cur_pd.landmarks;

      lands = LandMap3To2(bb_lands_fcsv);

      for (const auto& l : lands)
      {
        vout << fmt::sprintf("%s: [ %.4f , %.4f ]", l.first, l.second(0), l.second(1)) << std::endl;
      }

      pd.emplace_back(cur_pd);
    }
    else
    {
      vout << "    proj group not found... finished looking...." << std::endl;
      break;
    }
  }
  
  return pd;
}

std::vector<jhmr::tbme::ProjData>
jhmr::tbme::GetCamAndBBLands(const H5::Group& g, const H5::Group& pelvis_as_fid_parent_g,
                             std::ostream& vout, const bool include_anat_lands,
                             const bool include_pixels)
{
  auto pd = GetCamAndBBLands(g, vout, include_anat_lands, include_pixels);

  const CamModel orig_view_1_cam = pd[0].cam;

  vout << "recovering relative views using pelvis regis..." << std::endl;
  RecoverRelativeViewEncodingsPelvisFid(pd.begin(), pd.end(), orig_view_1_cam, pelvis_as_fid_parent_g);
  
  return pd;
}

jhmr::LandMap3 jhmr::tbme::TrianPts(const std::vector<ProjData>& pd)
{
  const size_type num_views = pd.size();

  std::vector<CamModel> cams(num_views);

  std::vector<std::decay<decltype(pd[0].landmarks)>::type> lands(num_views);

  for (size_type i = 0; i < num_views; ++i)
  {
    cams[i]  = pd[i].cam;
    lands[i] = pd[i].landmarks;
  }

  return Compute3DCamExtrinsPtsFrom2DViewsWithCorrMaps(cams, lands);
}

jhmr::PtList3 jhmr::tbme::SortAPPPtsByAP(const PtList3& src_pts_wrt_app, const size_type axis_idx)
{
  PtList3 sort_pts = src_pts_wrt_app;

  std::sort(sort_pts.begin(), sort_pts.end(),
            [axis_idx] (const Pt3& x, const Pt3& y)
            {
              return x(axis_idx) < y(axis_idx);
            });

  return sort_pts;
}

jhmr::Pt3 jhmr::tbme::FindAcetRimPt(const PtList3& rim_pts, const Pt3& fhc_wrt_app, const bool is_left)
{
  const size_type num_pts = rim_pts.size();

  jhmrASSERT(num_pts > 0);

  Plane3<CoordScalar> coronal_plane;
  coronal_plane.normal = { 0, 0, 1 };  // Out of plane direction is AP
  coronal_plane.scalar = coronal_plane.normal.dot(fhc_wrt_app);

  Pt3 cur_rim_pt = rim_pts[0];

  bool found_inter = false;

  for (size_type i = 0; i < (num_pts - 1); ++i)
  {
    const auto inter_info = LineSegPlaneIntersect(rim_pts[i], rim_pts[i+1], coronal_plane);
    
    if (std::get<0>(inter_info) >= 0)
    {
      const auto& p = std::get<1>(inter_info).inter_pt;
   
      // if this is the first intersection, then keep it, otherwise check against other intersections
      // for more maximum coverage about the LR axis. 
      if (!found_inter)
      {
        cur_rim_pt = p;
      }
      else
      { 
        if (is_left)
        {
          if (p(0) > cur_rim_pt(0))
          {
            cur_rim_pt = p;
          }
        }
        else
        {
          if (p(0) < cur_rim_pt(0))
          {
            cur_rim_pt = p;
          }
        }
      }
      
      found_inter = true;
    }
  }

  jhmrASSERT(found_inter);

  return cur_rim_pt;
}

jhmr::CoordScalar
jhmr::tbme::LateralCenterEdgeAngle(const Pt3& fhc_wrt_app, const Pt3& acet_rim_pt_wrt_app, const bool is_left)
{
  const Pt3 up_vec_app = { 0, 1, 0 };
 
  Pt3 v = acet_rim_pt_wrt_app - fhc_wrt_app;
  jhmrASSERT(std::abs(v(2)) < 1.0e-6);
  
  v /= v.norm();

  const CoordScalar s = is_left ? ((v(0) > 0) ? 1 : -1) : ((v(0) < 0) ? 1 : -1);

  return std::acos(v.dot(up_vec_app)) * s;
}

std::tuple<jhmr::CoordScalar,jhmr::Pt3,jhmr::Pt3>
jhmr::tbme::LateralCenterEdgeAngleAfterRepo(const FrameTransform& frag_xform,
                                            const Pt3& fhc_wrt_app,
                                            const PtList3& rim_pts_wrt_app, const bool is_left)
{
  const Pt3 fhc_repo = frag_xform * fhc_wrt_app;

  PtList3 rim_pts_repo(rim_pts_wrt_app.size());
  ApplyTransformToPts(frag_xform, rim_pts_wrt_app, &rim_pts_repo);

  const Pt3 rim_pt = FindAcetRimPt(rim_pts_repo, fhc_repo, is_left);

  const CoordScalar lce = LateralCenterEdgeAngle(fhc_repo, rim_pt, is_left);

  return std::make_tuple(lce, fhc_repo, rim_pt);
}

std::unordered_map<std::string,boost::any>& jhmr::tbme::GlobalDebugDataManager()
{
  static std::unordered_map<std::string,boost::any> m;

  return m;
}

jhmr::LandMap3 jhmr::tbme::TransformLandMap(const LandMap3& src, const FrameTransform& xform)
{
  LandMap3 dst;
  dst.reserve(src.size());

  for (const auto& v : src)
  {
    dst.insert(LandMap3::value_type(v.first, xform * v.second));
  }

  return dst;
}

namespace
{

// FROM:
//   https://git.lcsr.jhu.edu/bigss/osteotomy/raw/master/cpp/imageProcessing/radialSymmetry.h
//   https://git.lcsr.jhu.edu/bigss/osteotomy/raw/master/cpp/imageProcessing/radialSymmetry.cpp

/// Internal method to get numerical gradient for x components. 
/// @param[in] mat Specify input matrix.
/// @param[in] spacing Specify input space.
cv::Mat gradientX(cv::Mat & mat, double spacing = 1) {
    cv::Mat grad = cv::Mat::zeros(mat.cols,mat.rows,CV_64F);

    /*  last row */
    int maxCols = mat.cols;
    int maxRows = mat.rows;

    /* get gradients in each border */
    /* first row */
    cv::Mat col = (-mat.col(0) + mat.col(1))/(double)spacing;
    col.copyTo(grad(cv::Rect(0,0,1,maxRows)));

    col = (-mat.col(maxCols-2) + mat.col(maxCols-1))/(double)spacing;
    col.copyTo(grad(cv::Rect(maxCols-1,0,1,maxRows)));

    /* centered elements */
    cv::Mat centeredMat = mat(cv::Rect(0,0,maxCols-2,maxRows));
    cv::Mat offsetMat = mat(cv::Rect(2,0,maxCols-2,maxRows));
    cv::Mat resultCenteredMat = (-centeredMat + offsetMat)/(((double)spacing)*2.0);

    resultCenteredMat.copyTo(grad(cv::Rect(1,0,maxCols-2, maxRows)));
    return grad;
}

/// Internal method to get numerical gradient for y components. 
/// @param[in] mat Specify input matrix.
/// @param[in] spacing Specify input space.
cv::Mat gradientY(cv::Mat & mat, double spacing = 1) {

    cv::Mat grad = cv::Mat::zeros(mat.cols,mat.rows,CV_64F);

    /*  last row */
    const int maxCols = mat.cols;
    const int maxRows = mat.rows;

    /* get gradients in each border */
    /* first row */
    cv::Mat row = (-mat.row(0) + mat.row(1))/(double)spacing;
    row.copyTo(grad(cv::Rect(0,0,maxCols,1)));

    row = (-mat.row(maxRows-2) + mat.row(maxRows-1))/(double)spacing;
    row.copyTo(grad(cv::Rect(0,maxRows-1,maxCols,1)));

    /* centered elements */
    cv::Mat centeredMat = mat(cv::Rect(0,0,maxCols,maxRows-2));
    cv::Mat offsetMat = mat(cv::Rect(0,2,maxCols,maxRows-2));
    cv::Mat resultCenteredMat = (-centeredMat + offsetMat)/(((double)spacing)*2.0);

    resultCenteredMat.copyTo(grad(cv::Rect(0,1,maxCols, maxRows-2)));
    return grad;
}

std::pair<cv::Mat,cv::Mat> gradient(cv::Mat & img, double spaceX = 1, double spaceY = 1)
{
    cv::Mat gradY = gradientY(img,spaceY);
    cv::Mat gradX = gradientX(img,spaceX);
    std::pair<cv::Mat,cv::Mat> retValue(gradX,gradY);
    return retValue;
}

cv::Mat radialSymmetry(const cv::Mat &inputImg, std::vector< int > radius, double factor, int roi = 0)
{
  constexpr bool kDEBUG = false;

  // convert the input image to gray (single channel), 64-bit floats
  cv::Mat img;
  cv::Mat greyImage;
  if (inputImg.channels() >= 3)
    cv::cvtColor(inputImg, greyImage, cv::COLOR_RGB2GRAY);
  else
    greyImage = inputImg;
  greyImage.convertTo(img, CV_64F);

  // some variables
  cv::Mat imgx, imgy;
  cv::Mat mag(img.cols, img.rows, CV_64F);
  cv::Mat imgx2(img.cols, img.rows, CV_64F), imgy2(img.cols, img.rows, CV_64F);

  // get the image gradients
  std::pair<cv::Mat, cv::Mat> p = gradient(img);
  imgx = p.first;
  imgy = p.second;

  if (kDEBUG)
  {
    std::cout << "imgx = " << std::endl << imgx(cv::Range(0,5), cv::Range(0,3)) << std::endl;
    std::cout << "imgy = " << std::endl << imgy(cv::Range(0,5), cv::Range(0,3)) << std::endl;
  }

  // get the magnitude of the gradient at each pixel
  cv::pow(imgx, 2, imgx2);
  cv::pow(imgy, 2, imgy2);

  mag = imgx2 + imgy2;
  cv::sqrt(mag, mag);

  if (kDEBUG)
  {
    std::cout << "imgx2 = " << std::endl << imgx2(cv::Range(0,5), cv::Range(0,3)) << std::endl;
    std::cout << "imgy2 = " << std::endl << imgy2(cv::Range(0,5), cv::Range(0,3)) << std::endl;
    std::cout << "mag = " << mag(cv::Range(0,5), cv::Range(0,5)) << std::endl;
  }

  // normalize the gradients
  // division by 0 is automatically avoided by the divide function
  cv::divide(imgx, mag, imgx);
  cv::divide(imgy, mag, imgy);

  // Symmetry matrix
  cv::Mat S = cv::Mat::zeros(img.cols, img.rows, CV_64F);

  // create the meshgrids with pixel indices
  cv::Mat x, y;
  x = cv::Mat(imgx.cols, imgy.rows, CV_32S);
  y = cv::Mat(imgx.cols, imgy.rows, CV_32S);
  for(int i=0; i<imgx.cols; i++) {
    x.at<int>(0, i) = i;
    y.at<int>(i, 0) = i;
  }
  x = cv::repeat(x.row(0), img.cols, 1);
  y = cv::repeat(y.col(0), 1, img.rows);

  // some more variables
  cv::Mat posx, posy, negx, negy;
  cv::Mat mult(img.cols, img.rows, CV_64F);
  cv::Mat multR(img.cols, img.rows, CV_32S);

  // for each pixel radius
  for(std::vector<int>::iterator it = radius.begin(); it != radius.end(); ++ it) {

    int r = (*it);
    //cv::Mat posx(imgx), posy(imgy);

    // multiply gradients by radius, add to respective index
    cv::multiply(imgx, r, mult);
    mult.convertTo(multR, CV_32S);
    posx = x + multR;
    negx = x - multR;

    cv::multiply(imgy, r, mult);
    mult.convertTo(multR, CV_32S);
    posy = y + multR;
    negy = y - multR;

    // ensurve values are on [0 img.cols-1]
    cv::max(posx, 0, posx);
    cv::min(posx, img.cols-1, posx);
    cv::max(negx, 0, negx);
    cv::min(negx, img.cols-1, negx);

    // ensure values are on [0 img.rows-1]
    cv::max(posy, 0, posy);
    cv::min(posy, img.rows-1, posy);
    cv::max(negy, 0, negy);
    cv::min(negy, img.rows-1, negy);

    cv::Mat M = cv::Mat::zeros(img.cols, img.rows, CV_64F);
    cv::Mat W = cv::Mat::zeros(img.cols, img.rows, CV_64F);
    for(int i=0; i<img.cols; i++) {
      for(int j=0; j<img.rows; j++) {
        int yy = posx.at<int>(i, j);
        int xx = posy.at<int>(i, j);
        M.at<double>(xx, yy) += mag.at<double>(i, j);
        W.at<double>(xx, yy) += 1.0;

        yy = negx.at<int>(i, j);
        xx = negy.at<int>(i, j);
        M.at<double>(xx, yy) -= mag.at<double>(i, j);
        W.at<double>(xx, yy) -= 1.0;
      }
    }

    cv::Mat F(img.cols, img.rows, CV_64F);
    cv::multiply(M, cv::abs(W), F);
    cv::Mat filt(img.cols, img.rows, CV_64F);

    // create the gaussian kernel.
    // cannot use gaussian blur directly -- that requires an odd-sized aperture
    // we also want to modify the kernel through multiplying by r
    double start = -(r-1)/2;
    cv::Mat xr, yr;
    xr = cv::Mat(r, r, CV_64F);
    yr = cv::Mat(r, r, CV_64F);
    for(int i=0; i<r; i++) {
      xr.at<double>(0, i) = start;
      yr.at<double>(i, 0) = start;
      start++;
    }
    xr = cv::repeat(xr.row(0), r, 1);
    yr = cv::repeat(yr.col(0), 1, r);

    cv::Mat kernel;
    cv::exp(-(xr.mul(xr) + yr.mul(yr))/(2*0.25*0.25*r*r), kernel);
    double sum = cv::sum(kernel)[0];
    kernel /= sum;
    kernel *= r;

    cv::filter2D(F, filt, -1, kernel);
    S += filt;

#ifdef JHMR_HAS_OPENCV_HIGHGUI
    if (kDEBUG)
    {
      cv::imshow("Display", F);
      cv::waitKey(0);
      cv::imshow("Display", S);
      cv::waitKey(0);
    }
#endif
  }

  S = S / (double)radius.size();
  S = -S;
  cv::max(S, 0, S); // convert values less than 0 to 0
  double minVal, maxVal;

  // normalize
  cv::minMaxLoc(S, &minVal, &maxVal);
  S /= maxVal;

  // eliminate points out of the ROI
  if(roi > 0) {
    cv::Mat Sbinary = (S > 0);
    cv::Mat nonzeroS;
    cv::findNonZero(Sbinary, nonzeroS);
    double rCntr = img.rows / 2.0 - 6;
    double cCntr = img.cols / 2.0 - 6;
    int r, c;
    double dist;

   for(size_t i=0; i<nonzeroS.total(); i++) {
      r = nonzeroS.at<cv::Point>(i).x;
      c = nonzeroS.at<cv::Point>(i).y;
      dist = std::sqrt( (r-rCntr)*(r-rCntr) + (c-cCntr)*(c-cCntr) );
      if(dist > roi)
        S.at<double>(nonzeroS.at<cv::Point>(i)) = 0;
    }
  }

  // assume the first radius is the smallest
  int sze = 2*radius[0] + 1;
  cv::Mat mx;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(sze, sze));
  cv::dilate(S, mx, kernel);

#ifdef JHMR_HAS_OPENCV_HIGHGUI
  if (kDEBUG)
  {
    cv::imshow("Display", mx);
    cv::waitKey(0);
  }
#endif

  cv::minMaxLoc(S, &minVal, &maxVal);
  double thresh = factor * maxVal;

  cv::Mat eq = (S == mx); // sets equality to 255
  cv::Mat gtThresh = (S > thresh); // sets gt to 255

  cv::Mat valid;
  cv::bitwise_and(eq, gtThresh, valid);

  cv::Mat nonzero;
  cv::findNonZero(valid, nonzero);

#ifdef JHMR_HAS_OPENCV_HIGHGUI
  if (kDEBUG)
  {
    for(int i=0; i<nonzero.total(); i++) {
      int r = nonzero.at<cv::Point>(i).x;
      int c = nonzero.at<cv::Point>(i).y;
      std::cout << "(r, c) : (" << r << ", " << c << ")" << std::endl;
      cv::circle(inputImg, nonzero.at<cv::Point>(i), 3, cv::Scalar(0, 1, 0));
    }

    cv::imshow("Display", inputImg);
    cv::waitKey(0);
  }
#endif

  return nonzero;
}

}  // un-named

jhmr::PtList2 jhmr::tbme::DetectBBsIn2D(itk::Image<PixelScalar,2>* proj, const bool log_corrected,
                                        const bool find_large_bbs, const bool refine_w_max)
{
  cv::Mat proj_ocv = ShallowCopyItkToOpenCV(proj);

  // the radial symmetry algorithm appears to run best on pixel distributions where the BBs are dark
  // so flip the intensities for log corrected data. If we do not do this the approach fails to detect
  // any BBs

  cv::Mat tmp_ocv;

  if (log_corrected)
  {
    tmp_ocv = proj_ocv.clone();
    
    double tmp_min = 0;
    double tmp_max = 0;
    cv::minMaxLoc(tmp_ocv, &tmp_min, &tmp_max);

    tmp_ocv *= -1;
    tmp_ocv += tmp_max;
  }

  // settings for large and small BBs: radii { 1,2,3,4 }, factor=0.1
  // settings for small BBs: radii { 1,2 }, factor=0.2
  // settings for large BBs: radii { 4 }, factor=0.2

  const std::vector<int> radii = find_large_bbs ? std::vector<int>({ 4 }) : std::vector<int>({ 1,2 });

  cv::Mat bb_locs_mat = radialSymmetry(log_corrected ? tmp_ocv : proj_ocv, radii, 0.2);
  
  PtList2 bb_inds;

  const size_type num_bbs = bb_locs_mat.total();

  bb_inds.resize(num_bbs);

  for (size_type i = 0; i < num_bbs; ++i)
  {
    auto& p = bb_locs_mat.at<cv::Point>(i);

    bb_inds[i][0] = p.x;
    bb_inds[i][1] = p.y;
  }
 
  if (refine_w_max)
  {
    jhmrASSERT(log_corrected);

    // this interface takes a landmark map... create a temporary
    std::unordered_map<size_type,Pt2> tmp_inds;
    tmp_inds.reserve(num_bbs);
    
    for (size_type i = 0; i < num_bbs; ++i)
    {
      tmp_inds.emplace(i, bb_inds[i]);
    }

    RefineBB2DWithMaxIntens(proj, &tmp_inds, *std::max_element(radii.begin(), radii.end()));
    
    for (size_type i = 0; i < num_bbs; ++i)
    {
      bb_inds[i] = tmp_inds[i];
    }
  }

  return bb_inds;
}

bool jhmr::tbme::GetUseLargeBBs(const H5::CommonFG& h5)
{
  bool use_large_bbs = false;

  if (ObjectInGroupH5("bb-track-use-large-bbs", h5))
  {
    use_large_bbs = static_cast<bool>(ReadSingleScalarH5<long>("bb-track-use-large-bbs", h5));
  }

  return use_large_bbs;
}

jhmr::VolPtr jhmr::tbme::ConvertLabelVolToFloat(const LabelVol* lv)
{
  LabelVolPtr lv_copy = ITKImageDeepCopy(lv);

  VolPtr fv = CastITKImageIfNeeded<PixelScalar>(lv_copy.GetPointer());

  const auto vol_size = fv->GetLargestPossibleRegion().GetSize();

  const size_type num_vox = static_cast<size_type>(vol_size[0]) *
                            static_cast<size_type>(vol_size[1]) *
                            static_cast<size_type>(vol_size[2]);

  PixelScalar* buf = fv->GetBufferPointer();

  std::transform(buf, buf + num_vox, buf,
                 [] (const PixelScalar& p)
                 {
                   return (std::abs(p) > 1.0e-6) ? 1 : 0;
                 });

  return fv;
}

void jhmr::tbme::GTSeg2DHelper::init(GPUPrefsXML& gpu_prefs, const CamModel& cam)
{
  depth_rc.reset(new CameraRayCasterDepthOnlyGPU(gpu_prefs.ctx, gpu_prefs.queue));
  depth_rc->set_num_projs(1);
  depth_rc->set_render_thresh(0.5);
  depth_rc->set_ray_step_size(0.25);
  depth_rc->set_volumes(label_vols_as_float);
  depth_rc->use_proj_store_replace_method();
  
  depth_rc->set_camera_model(cam);
  
  depth_rc->allocate_resources();
   
  seg = MakeImageFromCam<LabelScalar>(cam);

  num_inters_img = cv::Mat(cam.num_det_rows, cam.num_det_cols, cv::DataType<PixelScalar>::type);
  bg_seg         = cv::Mat(cam.num_det_rows, cam.num_det_cols, cv::DataType<PixelScalar>::type);
}

void jhmr::tbme::GTSeg2DHelper::run(const std::unordered_map<std::string,FrameTransform>& vol_poses)
{
  bool has_left_hemi_pelvis_pose  = false;
  bool has_right_hemi_pelvis_pose = false;
  bool has_vertebrae_pose         = false;
  bool has_upper_sacrum_pose      = false;
  bool has_lower_sacrum_pose      = false;
  bool has_left_femur_pose        = false;
  bool has_right_femur_pose       = false;
  bool has_left_frag_pose         = false;
  bool has_right_frag_pose        = false;
  bool has_soft_tissue_pose       = false;
  
  cv::Mat& left_hemi_pelvis_depth_map  = left_hemi_pelvis_seg;
  cv::Mat& right_hemi_pelvis_depth_map = right_hemi_pelvis_seg;
  cv::Mat& vertebrae_depth_map         = vertebrae_seg;
  cv::Mat& upper_sacrum_depth_map      = upper_sacrum_seg;
  cv::Mat& lower_sacrum_depth_map      = lower_sacrum_seg;
  cv::Mat& left_femur_depth_map        = left_femur_seg;
  cv::Mat& right_femur_depth_map       = right_femur_seg;
  cv::Mat& left_frag_depth_map         = left_frag_seg;
  cv::Mat& right_frag_depth_map        = right_frag_seg;
  cv::Mat& soft_tissue_depth_map       = soft_tissue_seg;
 
  // reset these depth maps 
  left_hemi_pelvis_depth_map  = cv::Mat();
  right_hemi_pelvis_depth_map = cv::Mat();
  vertebrae_depth_map         = cv::Mat();
  upper_sacrum_depth_map      = cv::Mat();
  lower_sacrum_depth_map      = cv::Mat();
  left_femur_depth_map        = cv::Mat();
  right_femur_depth_map       = cv::Mat();
  left_frag_depth_map         = cv::Mat();
  right_frag_depth_map        = cv::Mat();
  soft_tissue_depth_map       = cv::Mat();

  constexpr PixelScalar kMAX_DIST = std::numeric_limits<PixelScalar>::max();
  
  auto convert_depth_to_seg = [] (cv::Mat& depth_map)
  {
    const int nr = depth_map.rows;
    const int nc = depth_map.cols;

    for (int r = 0; r < nr; ++r)
    {
      PixelScalar* cur_row = &depth_map.at<PixelScalar>(r,0);

      for (int c = 0; c < nc; ++c)
      {
        cur_row[c] = (cur_row[c] < kMAX_DIST) ? 1 : 0;
      }
    }
  };

  for (const auto& str_pose_pair : vol_poses)
  {
    const auto& vol_str = str_pose_pair.first;

    const auto vol_idx_it = vol_inds.find(vol_str);
    jhmrASSERT(vol_idx_it != vol_inds.end());
    
    depth_rc->xform_cam_to_itk_phys(0) = str_pose_pair.second;
    depth_rc->compute(vol_idx_it->second);
    
    cv::Mat cur_depth = depth_rc->proj_ocv(0).clone();
    
    if (vol_str == "soft-tissue")
    {
      has_soft_tissue_pose = true;
      soft_tissue_depth_map = cur_depth;
    }
    else if (vol_str == "left-hemi-pelvis")
    {
      has_left_hemi_pelvis_pose = true;
      left_hemi_pelvis_depth_map = cur_depth;
    }
    else if (vol_str == "right-hemi-pelvis")
    {
      has_right_hemi_pelvis_pose = true;
      right_hemi_pelvis_depth_map = cur_depth;
    }
    else if (vol_str == "vertebrae")
    {
      has_vertebrae_pose = true;
      vertebrae_depth_map = cur_depth;
    }
    else if (vol_str == "upper-sacrum")
    {
      has_upper_sacrum_pose = true;
      upper_sacrum_depth_map = cur_depth;
    }
    else if (vol_str == "lower-sacrum")
    {
      has_lower_sacrum_pose = true;
      lower_sacrum_depth_map = cur_depth;
    }
    else if (vol_str == "left-femur")
    {
      has_left_femur_pose = true;
      left_femur_depth_map = cur_depth;
    }
    else if (vol_str == "right-femur")
    {
      has_right_femur_pose = true;
      right_femur_depth_map = cur_depth;
    }
    else if (vol_str == "left-frag")
    {
      has_left_frag_pose = true;
      left_frag_depth_map = cur_depth;
    }
    else if (vol_str == "right-frag")
    {
      has_right_frag_pose = true;
      right_frag_depth_map = cur_depth;
    }
    else
    {
      jhmrThrow("Unsupported volume: %s", vol_str.c_str());
    }
  }
  
  seg->FillBuffer(0);
          
  cv::Mat seg_ocv = ShallowCopyItkToOpenCV(seg.GetPointer());

  const size_type nr = seg_ocv.rows;
  const size_type nc = seg_ocv.cols;
 
  for (int r = 0; r < nr; ++r)
  {
    const PixelScalar* left_femur_row        = has_left_femur_pose        ? &left_femur_depth_map.at<PixelScalar>(r,0)        : nullptr;
    const PixelScalar* right_femur_row       = has_right_femur_pose       ? &right_femur_depth_map.at<PixelScalar>(r,0)       : nullptr;
    const PixelScalar* left_frag_row         = has_left_frag_pose         ? &left_frag_depth_map.at<PixelScalar>(r,0)         : nullptr;
    const PixelScalar* right_frag_row        = has_right_frag_pose        ? &right_frag_depth_map.at<PixelScalar>(r,0)        : nullptr;
    const PixelScalar* left_hemi_pelvis_row  = has_left_hemi_pelvis_pose  ? &left_hemi_pelvis_depth_map.at<PixelScalar>(r,0)  : nullptr;
    const PixelScalar* right_hemi_pelvis_row = has_right_hemi_pelvis_pose ? &right_hemi_pelvis_depth_map.at<PixelScalar>(r,0) : nullptr;
    const PixelScalar* vertebrae_row         = has_vertebrae_pose         ? &vertebrae_depth_map.at<PixelScalar>(r,0)         : nullptr;
    const PixelScalar* upper_sacrum_row      = has_upper_sacrum_pose      ? &upper_sacrum_depth_map.at<PixelScalar>(r,0)      : nullptr;
    const PixelScalar* lower_sacrum_row      = has_lower_sacrum_pose      ? &lower_sacrum_depth_map.at<PixelScalar>(r,0)      : nullptr;
    const PixelScalar* soft_tissue_row       = has_soft_tissue_pose       ? &soft_tissue_depth_map.at<PixelScalar>(r,0)       : nullptr;

    LabelScalar* seg_row = &seg_ocv.at<LabelScalar>(r,0);

    for (int c = 0; c < nc; ++c)
    {
      LabelScalar& s = seg_row[c];

      if (has_left_femur_pose && (left_femur_row[c] < kMAX_DIST))
      {
        s = left_femur_label;
      }
      else if (has_right_femur_pose && (right_femur_row[c] < kMAX_DIST))
      {
        s = right_femur_label;
      }
      else if (has_left_frag_pose && (left_frag_row[c] < kMAX_DIST))
      {
        s = left_frag_label;
      }
      else if (has_right_frag_pose && (right_frag_row[c] < kMAX_DIST))
      {
        s = right_frag_label;
      }
      // TODO: handle cuts
      else
      {
        // Give preference to the hemi-pelves, if they are both present in this ray, choose the one with smaller depth
        
        const bool intersect_left_hemi_pelvis  = has_left_hemi_pelvis_pose  && (left_hemi_pelvis_row[c] < kMAX_DIST);
        const bool intersect_right_hemi_pelvis = has_right_hemi_pelvis_pose && (right_hemi_pelvis_row[c] < kMAX_DIST);

        if (intersect_left_hemi_pelvis && intersect_right_hemi_pelvis)
        {
          if (left_hemi_pelvis_row[c] < right_hemi_pelvis_row[c])
          {
            s = left_hemi_pelvis_label;
          }
          else
          {
            s = right_hemi_pelvis_label;
          }
        }
        else if (intersect_left_hemi_pelvis)
        {
          s = left_hemi_pelvis_label;
        }
        else if (intersect_right_hemi_pelvis)
        {
          s = right_hemi_pelvis_label;
        }
        else if (has_vertebrae_pose && (vertebrae_row[c] < kMAX_DIST))
        {
          s = vertebrae_label;
        }
        else if (has_upper_sacrum_pose && (upper_sacrum_row[c] < kMAX_DIST))
        {
          s = upper_sacrum_label;
        }
        else if (has_lower_sacrum_pose && (lower_sacrum_row[c] < kMAX_DIST))
        {
          s = lower_sacrum_label;
        }
        else if (has_soft_tissue_pose && (soft_tissue_row[c] < kMAX_DIST))
        {
          s = soft_tissue_label;
        }
        else
        {
          s = bg_label;
        }
      }
    }
  }
 
  if (true)
  {
    // First convert depth maps to segmentation masks
    // and compute the number of objects at each pixel

    num_inters_img.setTo(0);
    bg_seg.setTo(0);

    if (has_left_femur_pose)
    {
      convert_depth_to_seg(left_femur_depth_map);
      
      num_inters_img += left_femur_depth_map;
    }
    
    if (has_right_femur_pose)
    {
      convert_depth_to_seg(right_femur_depth_map);

      num_inters_img += right_femur_depth_map;
    }
    
    if (has_left_frag_pose)
    {
      convert_depth_to_seg(left_frag_depth_map);

      num_inters_img += left_frag_depth_map;
    } 
    
    if (has_right_frag_pose)
    {
      convert_depth_to_seg(right_frag_depth_map);

      num_inters_img += right_frag_depth_map;
    }
    
    if (has_left_hemi_pelvis_pose)
    {
      convert_depth_to_seg(left_hemi_pelvis_depth_map);

      num_inters_img += left_hemi_pelvis_depth_map;
    }
    
    if (has_right_hemi_pelvis_pose)
    {
      convert_depth_to_seg(right_hemi_pelvis_depth_map);
    
      num_inters_img += right_hemi_pelvis_depth_map;
    }
    
    if (has_vertebrae_pose)
    {
      convert_depth_to_seg(vertebrae_depth_map);

      num_inters_img += vertebrae_depth_map;
    }
    
    if (has_upper_sacrum_pose)
    {
      convert_depth_to_seg(upper_sacrum_depth_map);

      num_inters_img += upper_sacrum_depth_map;
    }

    if (has_lower_sacrum_pose)
    {
      convert_depth_to_seg(lower_sacrum_depth_map);

      num_inters_img += lower_sacrum_depth_map;
    }

    if (has_soft_tissue_pose)
    {
      convert_depth_to_seg(soft_tissue_depth_map);
      
      num_inters_img += soft_tissue_depth_map;
    }

    // Handle background (not intersecting anything we care about)
    for (int r = 0; r < nr; ++r)
    {
      const LabelScalar* seg_row = &seg_ocv.at<LabelScalar>(r,0);
      
      PixelScalar* num_inters_row = &num_inters_img.at<PixelScalar>(r,0);      
      
      PixelScalar* bg_seg_row = &bg_seg.at<PixelScalar>(r,0);

      for (int c = 0; c < nc; ++c)
      {
        if (seg_row[c] == bg_label)
        {
          jhmrASSERT(std::abs(num_inters_row[c]) < 1.0e-8);
          
          bg_seg_row[c]     = 1;
          num_inters_row[c] = 1;
        }
      }
    }

    // Next, convert the segmentation masks into [0,1] depending on
    // the number of objects at each pixel location
    
    bg_seg /= num_inters_img;

    if (has_left_femur_pose)
    {
      left_femur_depth_map /= num_inters_img;
    }
    
    if (has_right_femur_pose)
    {
      right_femur_depth_map /= num_inters_img;
    }
    
    if (has_left_frag_pose)
    {
      left_frag_depth_map /= num_inters_img;
    } 
    
    if (has_right_frag_pose)
    {
      right_frag_depth_map /= num_inters_img;
    }
    
    if (has_left_hemi_pelvis_pose)
    {
      left_hemi_pelvis_depth_map /= num_inters_img;
    }
    
    if (has_right_hemi_pelvis_pose)
    {
      right_hemi_pelvis_depth_map /= num_inters_img;
    }
    
    if (has_vertebrae_pose)
    {
      vertebrae_depth_map /= num_inters_img;
    }
    
    if (has_upper_sacrum_pose)
    {
      upper_sacrum_depth_map /= num_inters_img;
    }

    if (has_lower_sacrum_pose)
    {
      lower_sacrum_depth_map /= num_inters_img;
    }

    if (has_soft_tissue_pose)
    {
      soft_tissue_depth_map /= num_inters_img;
    }
  }
}

jhmr::tbme::CamModel jhmr::tbme::ReadCommonCamModel(const H5::CommonFG& h5)
{
  H5::Group cam_g = h5.openGroup("proj-params");

  CamModel cam_model;

  cam_model.coord_frame_type = CamModel::kORIGIN_AT_FOCAL_PT_DET_NEG_Z;

  cam_model.setup(ReadMatrixH5<CoordScalar>("intrinsic", cam_g),
                  Mat4x4(ReadAffineTransform4x4H5<CoordScalar>("extrinsic", cam_g).matrix()),
                  ReadSingleScalarH5<size_type>("num-rows", cam_g),
                  ReadSingleScalarH5<size_type>("num-cols", cam_g),
                  ReadSingleScalarH5<CoordScalar>("pixel-row-spacing", cam_g),
                  ReadSingleScalarH5<CoordScalar>("pixel-col-spacing", cam_g));
  
  return cam_model;
}

