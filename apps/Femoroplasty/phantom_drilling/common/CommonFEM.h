
#ifndef COMMONFEM_H_
#define COMMONFEM_H_

#include "jhmrCommon.h"

#include "jhmrProgOptUtils.h"
#include "jhmrProjData.h"
#include "jhmrHDF5.h"
#include "jhmrCameraRayCastingGPU.h"
#include "jhmrPAOUtils.h"
#include "jhmrMultiObjMultiLevel2D3DRegi.h"
#include "jhmrCountdown.h"

// Boost forward declarations
namespace boost
{
class any;
}  // boost

namespace jhmr
{

// Forward Declarations
struct GPUPrefsXML;

namespace fem
{

using ProjData       = SingleProjData<PixelScalar>;
using CamModel       = ProjData::CamModel;

using CutLabelProjData = SingleProjData<LabelScalar>;

using RayCaster           = jhmr::CameraRayCaster<PixelScalar,CoordScalar>;
using RayCasterLineIntGPU = jhmr::CameraRayCasterGPULineIntegral;

using PAOCuts     = PAOCutPlaneDefs<Pt3>;
using PAODispInfo = PAOCutDispInfo<Pt3>;

//constexpr size_type kNUM_RAND_CUTS            = 15;
//constexpr size_type kNUM_REPOS_PER_CUT        = 20;
//constexpr size_type kNUM_INIT_GUESSES         = 5;

struct PelvisFragFemurLabels
{
  LabelScalar pelvis_label;
  LabelScalar frag_label;
  LabelScalar femur_label;
  LabelScalar contra_femur_label;
};

struct FragRegiParams
{
  using Regi2D3D = MultiLevelMultiObjRegi::Regi2D3D;
  using RegFn    = Regi2D3D::PenaltyFnPtr;

  std::function<std::shared_ptr<MultiLevelMultiObjRegi::SimMetric>(
                                  const boost::compute::context&,
                                  const boost::compute::command_queue&,
                                  const FragRegiParams&,
                                  const size_type)> get_sim_metric_obj;

  // Level 1:
  
  double lvl_1_ds_factor;

  boost::optional<size_type> lvl_1_smooth_kernel_size;

  size_type lvl_1_pelvis_pop_size;
  std::vector<double> lvl_1_pelvis_sigma;
  std::vector<double> lvl_1_pelvis_bounds;
  double lvl_1_pelvis_tol_fn;
  double lvl_1_pelvis_tol_x;
  RegFn lvl_1_pelvis_reg_fn;
  double lvl_1_pelvis_reg_img_coeff;
  
  std::function<std::function<void()>(Regi2D3D*)> get_lvl_1_pelvis_before_regi_callback;

  size_type lvl_1_femur_pop_size;
  std::vector<double> lvl_1_femur_sigma;
  std::vector<double> lvl_1_femur_bounds;
  double lvl_1_femur_tol_fn;
  double lvl_1_femur_tol_x;
  RegFn lvl_1_femur_reg_fn;
  double lvl_1_femur_reg_img_coeff;
  
  std::function<std::function<void()>(Regi2D3D*)> get_lvl_1_femur_before_regi_callback;

  size_type lvl_1_frag_pop_size;
  std::vector<double> lvl_1_frag_sigma;
  std::vector<double> lvl_1_frag_bounds;
  double lvl_1_frag_tol_fn;
  double lvl_1_frag_tol_x;
  RegFn lvl_1_frag_reg_fn;
  double lvl_1_frag_reg_img_coeff;
  
  std::function<std::function<void()>(Regi2D3D*)> get_lvl_1_frag_before_regi_callback;
 
  // Level 2:

  double lvl_2_ds_factor;
  
  boost::optional<size_type> lvl_2_smooth_kernel_size;

  std::vector<double> lvl_2_pelvis_bounds;
  double lvl_2_pelvis_tol_fn;
  double lvl_2_pelvis_tol_x;
  
  std::function<std::function<void()>(Regi2D3D*)> get_lvl_2_pelvis_before_regi_callback;

  std::vector<double> lvl_2_femur_bounds;
  double lvl_2_femur_tol_fn;
  double lvl_2_femur_tol_x;
  
  std::function<std::function<void()>(Regi2D3D*)> get_lvl_2_femur_before_regi_callback;

  std::vector<double> lvl_2_frag_bounds;
  double lvl_2_frag_tol_fn;
  double lvl_2_frag_tol_x;
  
  std::function<std::function<void()>(Regi2D3D*)> get_lvl_2_frag_before_regi_callback;

  std::vector<double> lvl_2_all_bounds;
  double lvl_2_all_tol_fn;
  double lvl_2_all_tol_x;
  
  std::function<std::function<void()>(Regi2D3D*)> get_lvl_2_all_before_regi_callback;
};

FragRegiParams MakeFragRegiParamsOldSim(const bool use_patch_sim);

FragRegiParams MakeFragRegiParamsNewSim(const bool use_patch_sim);

FragRegiParams MakeFragRegiParamsNewSimPelvisFid(const bool use_patch_sim);

FragRegiParams MakeFragRegiParamsNewSimReg(const bool use_patch_sim);

FragRegiParams MakeFragRegiParamsNewSimPelvisFidReg(const bool use_patch_sim);

FragRegiParams GetFragRegiParamsUsingCmdLineArgs(const bool use_old_sim_params,
                                                 const bool use_known_rel_views,
                                                 const bool do_est_pu_il_regi,
                                                 const bool use_patch_sim,
                                                 const bool use_reg);

// pop size of 100 was the original value used here
FragRegiParams MakePelvisRegiParamsRegAPView(const bool use_patch_sim, const size_type pop_size = 100);

FragRegiParams MakePelvisRegiParamsRegRotFromAPView(const bool use_patch_sim, const size_type pop_size = 20);

struct CommonProgArgs
{
  bool verbose = false;

  std::string data_name = "soft-tissue";
  
  boost::optional<size_type> single_cut_idx;

  boost::optional<size_type> single_repo_idx;

  boost::optional<size_type> single_init_idx;

  std::string tmp_dir_root =
#ifdef __APPLE__
    // use external scratch drive
    "/Volumes/scratch2/zzztmp";
#else
    // this is on MARCC
    "/home-4/rgrupp3@jhu.edu/scratch/pao_tmp";
#endif

  std::ostream& vout();

  std::shared_ptr<std::ostream> null_vout;
};

void SetupCommonProgArgs(ProgOpts* po, const CommonProgArgs& args = CommonProgArgs());

CommonProgArgs GetCommonProgArgs(const ProgOpts& po);

Pt3 GetFHMidPtWrtVol(const H5::CommonFG& h5);

void SetupMaskProgArgs(ProgOpts* po, const std::string& default_name = "");

std::string GetMaskProgArgs(const ProgOpts& po);

std::string GetFragRegiNameUsingMaskStr(const std::string frag_regi_str, const std::string& mask_arg);

std::string GetMaskProjGroupName(const std::string mask_arg);

size_type MapPatientIDToIndex(const std::string& pat_id);

std::string GetPatientID(const H5::CommonFG& h5);

std::string GetSideString(const H5::CommonFG& h5);

bool IsLeftSide(const H5::CommonFG& h5);

std::string GetAnatLandmarksMapH5Path();

LandMap3 GetAnatLandmarksMap(const H5::CommonFG& h5);

FrameTransform GetAPPToVolStdOrigin(const H5::CommonFG& h5);

FrameTransform GetAPPToVolFHOrigin(const H5::CommonFG& h5);

std::array<Pt3,2> GetBothFHWrtVol(const H5::CommonFG& h5);

Pt3 GetIpsilFHPtWrtVol(const H5::CommonFG& h5);

PelvisFragFemurLabels GetPelvisFragFemurLabels(const H5::CommonFG& h5, const LabelVol* seg);

std::string GetOrigIntensVolH5Path();

std::string GetPelvisFemursSegH5Path();

std::string GetGTCutsSegH5Path();

std::string GetGTCutsDefH5Path();

std::string GetActualCutsSegH5Path(const size_type cut_idx);

std::string GetActualCutsDefH5Path(const size_type cut_idx);

std::string GetKnownPubisIliumCutsSegH5Path(const size_type cut_idx);

std::string GetProjDataPath(const size_type cut_idx,
                            const size_type repo_idx,
                            const std::string& data_name);

VolPtr GetOrigIntensVol(const H5::CommonFG& h5);

VolPtr GetOrigIntensVolAsLinAtt(const H5::CommonFG& h5);

LabelVolPtr GetPelvisSeg(const H5::CommonFG& h5);

LabelVolPtr GetGTCutsSeg(const H5::CommonFG& h5);

H5::Group GetRandPlansParentGroup(const H5::CommonFG& h5);

H5::Group GetReposParentGroup(const H5::CommonFG& h5);

std::vector<H5::Group> GetNumberedGroups(const H5::CommonFG& h5);

std::vector<H5::Group> GetSubsetOfOneNumberedGroup(const std::vector<H5::Group>& gs, const size_type idx);

std::vector<FrameTransform> GetInitCamWrtVols(const H5::CommonFG& h5);

std::string GetFullProjPathH5(const size_type cut_idx, const size_type repo_idx, const std::string data_name);

std::string GetFullRegiPathH5(const size_type cut_idx, const size_type repo_idx, const std::string data_name,
                              const std::string frag_regi_str,
                              const size_type init_idx, const bool use_known_rel_views = true);

std::string GetFullPelvisRegiPathH5(const size_type cut_idx, const size_type repo_idx, const std::string data_name,
                                    const std::string frag_regi_str,
                                    const size_type init_idx, const bool use_known_rel_views = true,
                                    const int rel_view_idx = -1);

std::string GetFullRegiDebugPathH5(const size_type cut_idx, const size_type repo_idx, const std::string data_name,
                                   const std::string frag_regi_str,
                                   const size_type init_idx, const bool use_known_rel_views = true);

std::string GetFullPelvisRegiDebugPathH5(const size_type cut_idx, const size_type repo_idx,
                                         const std::string data_name,
                                         const std::string frag_regi_str,
                                         const size_type init_idx,
                                         const bool use_known_rel_views = true,
                                         const int rel_view_idx = -1);

void SetupMultiObjMultiLevelFragRegi(MultiLevelMultiObjRegi* ml_mo_regi, GPUPrefsXML* gpu_prefs,
                                     const FragRegiParams& params, std::ostream& vout);

void SetupPelvisMultiLevelRegi(MultiLevelMultiObjRegi* ml_mo_regi, GPUPrefsXML* gpu_prefs,
                               const FragRegiParams& params, std::ostream& vout);

void SetupPelvisMultiLevelRegiSingleAPView(MultiLevelMultiObjRegi* ml_mo_regi, GPUPrefsXML* gpu_prefs,
                                           const FragRegiParams& params, std::ostream& vout);

void SetupPelvisMultiLevelRegiSingleRotFromAPView(MultiLevelMultiObjRegi* ml_mo_regi, GPUPrefsXML* gpu_prefs,
                                                  const FragRegiParams& params,
                                                  const FrameTransform& prev_view_pelvis_pose,
                                                  std::ostream& vout);

cv::Mat OverlayLabelMapAsRGBOnGrayscale(const cv::Mat& gray_img,
                                        const itk::Image<LabelScalar,2>* labels,
                                        const double mask_alpha = 0.8);

void UpdateCountdownH5(Countdown* time_limit, H5::H5File& h5, std::ostream& vout);

std::unique_ptr<Countdown> MakeCountdown(const std::string& time_format,
                                         H5::H5File& h5,
                                         std::ostream& vout);

// This is a template so it can be used for intensity projection data
// and cut labels projection data
template <class tIter>
void RemoveRelativeViewEncodings(tIter proj_data_begin, tIter proj_data_end,
                                 const CamModel& cam)
{
  using Iter = tIter;
  
  for (Iter it = proj_data_begin; it != proj_data_end; ++it)
  {
    it->cam = cam;
  }
}

template <class tIter>
void RemoveRelativeViewEncodings(tIter proj_data_begin, tIter proj_data_end)
{
  using Iter = tIter;

  Iter it = proj_data_begin;
  
  const auto& view_1_cam = it->cam;

  ++it;
  
  for (; it != proj_data_end; ++it)
  {
    it->cam = view_1_cam;
  }
}

template <class tIter>
void RecoverRelativeViewEncodingsPelvisFid(tIter proj_data_begin, tIter proj_data_end,
                                           const CamModel& view_1_cam,
                                           const H5::Group& h5)
{
  using Iter = tIter;

  RemoveRelativeViewEncodings(proj_data_begin, proj_data_end, view_1_cam);

  std::vector<CamModel> orig_cams;
  for (Iter proj_it = proj_data_begin; proj_it != proj_data_end; ++proj_it)
  {
    orig_cams.push_back(proj_it->cam);
  }

  const size_type num_views = orig_cams.size();

  std::vector<FrameTransform> pelvis_poses;

  for (size_type i = 0; i < num_views; ++i)
  {
    pelvis_poses.push_back(ReadAffineTransform4x4H5<CoordScalar>("regi-cam-to-pelvis-vol",
                                                                 h5.openGroup(fmt::sprintf("pelvis-regi-view-%lu", i))));
  }

  const auto new_cams = CreateCameraWorldUsingFiducial(orig_cams, pelvis_poses);

  auto cam_it = new_cams.begin();

  for (Iter proj_it = proj_data_begin; proj_it != proj_data_end; ++proj_it, ++cam_it)
  {
    proj_it->cam = *cam_it;
  }
}

class EstCutsAndCreateFragHelper : public Algorithm
{
public:
  std::string side_str;

  // these are defaulted to values used in the simulation study
  // index 0 is AP view, index 2 has detector more towards the contralateral side  
  size_type il_proj_idx = 2;
  size_type pu_proj_idx = 0;
  
  bool no_il_exit       = false;
  bool no_il_entry      = false;
  bool no_il_exit_entry = false;
  bool no_pu_exit       = false;
  bool no_pu_entry      = false;
  bool no_pu_exit_entry = false; 

  bool print_cmds = false;

  EstCutsAndCreateFragHelper(const std::string& est_cuts_exe_path,
                             const std::string& create_frag_seg_exe_path,
                             const std::string& work_dir = ".");

  void set_full_pelvis_seg(const LabelVol* seg);

  void set_hemipelvis_scalar_seg(const Vol* hemi);

  void set_invalid_frag_mask(const LabelVol* mask);

  void set_anat_lands(const LandMap3& lands);

  void set_regi(const FrameTransform& regi_xform);

  void set_orig_cuts(const PAOCuts& cuts_def, const PAODispInfo& cuts_disp);

  void set_est_cuts(const PAOCuts& cuts_def, const PAODispInfo& cuts_disp);

  void set_proj_labels(const std::vector<CutLabelProjData>& proj_labels);

  std::tuple<LabelVolPtr,double> create_frag_seg() const;

  std::tuple<PAOCuts,PAODispInfo,PtList3,PtList3,double> est_cuts() const;

private:

  std::string verbose_arg_str() const;

  double get_time_from_log() const;

  const std::string est_cuts_exe_path_;
  const std::string create_frag_seg_exe_path_;

  const std::string work_dir_;

  const std::string tmp_full_pelvis_seg_path_;
  const std::string tmp_hemi_pelvis_seg_path_;
  const std::string tmp_final_cuts_seg_path_;
  const std::string tmp_final_cuts_xml_path_;
  const std::string tmp_init_cuts_xml_path_;
  const std::string tmp_lands_path_;
  const std::string tmp_regi_pose_path_;
  const std::string tmp_label_projs_path_;
  const std::string tmp_invalid_cuts_path_;

  const std::string tmp_il_recon_pts_fcsv_path_;
  const std::string tmp_pu_recon_pts_fcsv_path_;
  
  const std::string tmp_log_file_path_;
};

FrameTransform CorrPtCloudRegi(const LandMap3& dst_lands_map,
                               const LandMap3& src_lands_map);

LandMap3 ExtractAllIliumPts(const LandMap3& src);

LandMap3 ExtractRightIliumPts(const LandMap3& src);

LandMap3 ExtractLeftIliumPts(const LandMap3& src);

LandMap3 ExtractIliumPtsSided(const LandMap3& src, const bool is_left);

LandMap3 ExtractAllFragPts(const LandMap3& src);

LandMap3 ExtractRightFragPts(const LandMap3& src);

LandMap3 ExtractLeftFragPts(const LandMap3& src);

LandMap3 ExtractFragPtsSided(const LandMap3& src, const bool is_left);

LandMap3 ExtractPtsSided(const LandMap3& src, const bool is_left);

LandMap3 ExtractSmallBBs(const LandMap3& src);

LandMap3 ExtractLargeBBs(const LandMap3& src);

std::unordered_map<std::string,Eigen::Matrix<double,3,1>> LandMapDouble(const LandMap3& f);

std::unordered_map<std::string,Eigen::Matrix<double,2,1>> LandMapDouble(const LandMap2& f);

LandMap2 LandMap3To2(const LandMap3& src);

// This version does NOT recover the relative poses
std::vector<ProjData> GetCamAndBBLands(const H5::Group& g, std::ostream& vout,
                                       const bool include_anat_lands = false,
                                       const bool include_pixels = false);

// This version recovers the relative poses of the c-arm
std::vector<ProjData> GetCamAndBBLands(const H5::Group& g, const H5::Group& pelvis_as_fid_parent_g,
                                       std::ostream& vout, const bool include_anat_lands = false,
                                       const bool include_pixels = false);

LandMap3 TrianPts(const std::vector<ProjData>& pd);

// axis_idx = 2 for APP, = 1 for LPS
PtList3 SortAPPPtsByAP(const PtList3& src_pts_wrt_app, const size_type axis_idx = 2);

Pt3 FindAcetRimPt(const PtList3& rim_pts, const Pt3& fhc_wrt_app, const bool is_left);

CoordScalar LateralCenterEdgeAngle(const Pt3& fhc_wrt_app, const Pt3& acet_rim_pt_wrt_app, const bool is_left);

// LCE angle, FH wrt app after frag xform, rim point wrt app after frag xform
std::tuple<CoordScalar,Pt3,Pt3>
LateralCenterEdgeAngleAfterRepo(const FrameTransform& frag_xform,
                                const Pt3& fhc_wrt_app,
                                const PtList3& rim_pts_wrt_app, const bool is_left);

std::unordered_map<std::string,boost::any>& GlobalDebugDataManager();

LandMap3 TransformLandMap(const LandMap3& src, const FrameTransform& xform);

PtList2 DetectBBsIn2D(itk::Image<PixelScalar,2>* proj, const bool log_corrected = false,
                      const bool find_large_bbs = false);

bool GetUseLargeBBs(const H5::CommonFG& h5);

}  // fem
}  // jhmr

#endif

