/*
 * MIT License
 *
 * Copyright (c) 2020 Robert Grupp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "xregIntensity2D3DRegiCMAES-JustinSnake.h"

#include <fmt/format.h>

#include <opencv2/imgcodecs.hpp>

#include <cmaes_interface.h>

#include "xregIntensity2D3DRegiDebug.h"
#include "xregSE3OptVars.h"
#include "xregRigidUtils.h"
#include "xregRotUtils.h"
#include "xregHDF5.h"
#include "xregImgSimMetric2D.h"
#include "xregFilesystemUtils.h"
#include "xregITKIOUtils.h"
#include "xregITKOpenCVUtils.h"
#include "xregOpenCVUtils.h"
#include "spline.h"

using namespace xreg;
using ListofTransformVec = std::vector<std::vector<double>>;
using TransformVec = std::vector<double>;

xreg::Intensity2D3DRegiCMAESjustinsnake::Intensity2D3DRegiCMAESjustinsnake()
{
  this->num_projs_per_view_ = kDEFAULT_LAMBDA;

  set_opt_obj_fn_tol(1.0e-3);
  set_opt_x_tol(1.0e-6);
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_pop_size(const size_type pop_size)
{
  this->num_projs_per_view_ = pop_size;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_bounds(const ScalarList& bounds)
{
  bounds_ = bounds;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::remove_bounds()
{
  bounds_.clear();
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_sigma(const ScalarList& sigma)
{
  sigma_ = sigma;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::reset_sigma()
{
  sigma_.clear();
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_Yctr_save_path(const std::string Yctr_save_path)
{
  Yctr_save_path_ = Yctr_save_path;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_ctr_mean(const ScalarList& ctr_mean)
{
  ctr_mean_ = ctr_mean;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_notch_rot_cen_list(const Pt3List& notch_rot_cen_list)
{
  notch_rot_cen_list_ = notch_rot_cen_list;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_notch_ref_xform_list(const FrameTransformList& notch_ref_xform_list)
{
  notch_ref_xform_list_ = notch_ref_xform_list;
}

/// \brief Convert CMAES optimization parameters to snake notch xform vectors following
ListofTransformVec xreg::Intensity2D3DRegiCMAESjustinsnake::Convert_optparams_to_snake_xformvec(const double* opt_params)
{
  ListofTransformVec snake_xformvec;

  const size_type num_vols = this->num_vols();

  const size_type num_params_per_xform  = this->opt_vars_->num_params();

  // transfer to intermediate storage:
  tk::spline sp_interp;
  std::vector<double> X_ctr(5), Y_ctr;

  for(size_type idx = 0; idx < 5; idx++){
    X_ctr[idx] = double(26.0*(idx+1)/6.0);
  }

  Pt6 notch_pt6, base_pt6;
  TransformVec base_vec(6), notch_vec(6);
  Mat4x4 Inter_Mat;

  const size_type snake_paramoff = 6;
  const size_type ctr_paramoff = snake_paramoff + 5;

  base_vec.assign(opt_params, opt_params + snake_paramoff);
  snake_xformvec.push_back(base_vec);

  Y_ctr.assign(opt_params + snake_paramoff, opt_params + ctr_paramoff);

  for (size_type indy = 0; indy < 5; ++indy){
    Y_ctr[indy] += ctr_mean_[indy];
  }

  sp_interp.set_points(X_ctr, Y_ctr);

  // then do notch volume:
  double rotZ = 0.0;
  Float accumX = 0.;
  Float accumY = 0.;
  Float transX = 0.;
  Float transY = 0.;
  for(size_type idx = 0; idx < 6; ++idx){
    base_pt6[idx] = opt_params[idx];
  }
  Inter_Mat = ExpSE3(base_pt6);

  FrameTransform src_wrt_base_delta, seg_wrt_src, seg_wrt_base;
  src_wrt_base_delta.matrix() = Inter_Mat;
  const FrameTransform src_wrt_base = this->regi_xform_guesses_[0] * this->intermediate_frames_[0] * src_wrt_base_delta * this->intermediate_frames_[0].inverse();

  for(size_type vol_idx = 1; vol_idx < num_vols; ++vol_idx)
  {
    // this->intermediate_frames_[vol_idx] = FrameTransform::Identity();
    rotZ += sp_interp(vol_idx-1);

    Float dist_rot_cen = vol_idx > 1 ? (notch_rot_cen_list_[vol_idx-1] - notch_rot_cen_list_[vol_idx-2]).norm() : 0.;//TODO: rot center list index
    accumX += dist_rot_cen * sin(rotZ * kDEG2RAD);
    accumY += dist_rot_cen * cos(rotZ * kDEG2RAD);
    Float rotcenX = notch_rot_cen_list_[vol_idx-1][0] - notch_rot_cen_list_[0][0];
    Float rotcenY = notch_rot_cen_list_[vol_idx-1][1] - notch_rot_cen_list_[0][1];
    Float accumX_wrt_rotcen = accumX - rotcenX;
    Float accumY_wrt_rotcen = accumY - rotcenY;
    transX = accumY_wrt_rotcen * sin(rotZ * kDEG2RAD) - accumX_wrt_rotcen * cos(rotZ * kDEG2RAD);
    transY = -(accumX_wrt_rotcen * sin(rotZ * kDEG2RAD) + accumY_wrt_rotcen * cos(rotZ * kDEG2RAD));
    transX = vol_idx % 2 == 0 ? transX + 0.03 : transX - 0.03;

    FrameTransform notch_rotation_xform = EulerRotXYZTransXYZFrame(0, 0, rotZ * kDEG2RAD, transX, transY, 0);

    FrameTransform cur_ref_xform = notch_ref_xform_list_[vol_idx];
    FrameTransform cur_vol_xform = cur_ref_xform.inverse() * notch_rotation_xform * cur_ref_xform * src_wrt_base;
    FrameTransform cur_vol_delta_wrt_initguess_xform = this->intermediate_frames_[vol_idx].inverse() * this->regi_xform_guesses_[vol_idx].inverse() * cur_vol_xform * this->intermediate_frames_[vol_idx];

    notch_pt6 = ExpRigid4x4ToPt6(cur_vol_delta_wrt_initguess_xform.matrix());

    // transfer to pop_params after calculation
    notch_vec.clear();
    for(size_type idx = 0; idx < 6; ++idx)
      notch_vec.push_back(notch_pt6[idx]);

    snake_xformvec.push_back(notch_vec);
  }

  return snake_xformvec;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::run()
{
  // TODO: support optimizing over multiple sources

  constexpr Scalar kDEFAULT_SIGMA = 0.3;  ///< Default sigma value to be used in all directions

  const auto& opt_vars = *this->opt_vars_;

  // number of volumes we're trying to estimate the pose of.
  const size_type nv = this->num_vols();

  const size_type num_params_per_xform = opt_vars.num_params();

  const size_type tot_num_params = 11;

  const size_type pop_size = this->num_projs_per_view_;

  xregASSERT(sigma_.empty() || (sigma_.size() == tot_num_params));
  xregASSERT(bounds_.empty() || (bounds_.size() == tot_num_params));

  if (sigma_.empty())
  {
    sigma_.assign(tot_num_params, kDEFAULT_SIGMA);
  }

  const bool run_unc = bounds_.empty();

  // setup the intermediate structure to store projection parameterizations:
  // pop_params[i][j][k] is the kth parameter of the jth SE(3) (projection) element for the ith object/volume
  ListOfListsOfScalarLists pop_params(nv, ListOfScalarLists(pop_size, ScalarList(num_params_per_xform, 0)));

  // intermediate structure to store the similarity values
  ScalarList sim_vals(pop_size, 0);

  std::vector<double> init_delta_guess(tot_num_params, 0);
  std::vector<double> tmp_sigma_double(sigma_.begin(), sigma_.end());

  evo_ = std::make_shared<cmaes_t>();

  auto* evo = evo_.get();

  memset(evo, 0, sizeof(cmaes_t));

  // The return value will store the objective function values at each of the
  // lambda search points
  // The "non" string parameter indicates that no parameter file should be read
  // or written to.
  double* obj_fn_vals = cmaes_init(evo, static_cast<int>(tot_num_params), &init_delta_guess[0],
                                   &tmp_sigma_double[0], 0, static_cast<int>(pop_size), "non");

  evo->sp.stopTolFun = obj_fn_tol_;

  evo->sp.stopTolX = x_tol_;

  this->before_first_iteration();

  OptAux* opt_aux = nullptr;

  // Turning aux output off, since it takes up a significant amount of space in HDF5
  // for the PAO simulation studies
  if (false && this->debug_save_iter_debug_info_)
  {
    this->debug_info_->opt_aux = std::make_shared<OptAux>();
    opt_aux = static_cast<OptAux*>(this->debug_info_->opt_aux.get());

    opt_aux->se3_param_dim = num_params_per_xform;
    opt_aux->pop_size = pop_size;

    const size_type init_capacity = this->debug_info_->sims.capacity();
    opt_aux->cov_mats.reserve(init_capacity);
    opt_aux->num_rejects.reserve(init_capacity);
    opt_aux->sim_vals.reserve(init_capacity);
    opt_aux->pop_params.reserve(init_capacity);
  }

  debug_sim_val_ = std::numeric_limits<Scalar>::max();

  size_type iter = 0;

  while (!cmaes_TestForTermination(evo) && (iter < this->max_num_iters_))
  {
    this->begin_of_iteration(get_cmaes_cur_params_list());

    // generate lambda new search points, sample population
    // This is an array of double arrays, e.g. pop[i] has tot_num_params doubles.
    double* const* pop = cmaes_SamplePopulation(evo);  // do not change content of pop

    if (!run_unc)
    {
      size_type num_rejects = 0;

      // enforce constraints
      for (size_type pop_ind = 0; pop_ind < pop_size; ++pop_ind)
      {
        bool cur_params_in_bounds = true;

        do
        {
          cur_params_in_bounds = true;

          const double* cur_params = pop[pop_ind];

          for (size_type param_idx = 0; param_idx < tot_num_params; ++param_idx)
          {
            const double cur_bound = bounds_[param_idx];

            // if out of bounds, then resample and re-check the newly sampled version
            if ((cur_params[param_idx] < -cur_bound) || (cur_params[param_idx] > cur_bound))
            {
              cmaes_ReSampleSingle(evo, pop_ind);
              cur_params_in_bounds = false;
              ++num_rejects;
              break;
            }
          }
        }
        while (!cur_params_in_bounds);
      }

      if (opt_aux)
      {
        opt_aux->num_rejects.push_back(num_rejects);
      }
    }

    // transfer to intermediate storage:
    for (size_type pop_ind = 0; pop_ind < pop_size; ++pop_ind)
    {
      // Convert each sampled CMAES population parameter to snake notch pose vector
      auto snake_xform_vec = Convert_optparams_to_snake_xformvec(pop[pop_ind]);
      for (size_type vol_idx = 0; vol_idx < nv; ++vol_idx)
      {
        // Copy each notch pose vector to population parameters for objective function computation
        pop_params[vol_idx][pop_ind].assign(snake_xform_vec[vol_idx].begin(), snake_xform_vec[vol_idx].end());
      }
    }

    // compute the DRRs and similarities
    // this->snake_obj_fn(pop_params, &sim_vals, nv);
    this->obj_fn(pop_params, &sim_vals);

    // copy similarities from intermediate storage
    std::copy(sim_vals.begin(), sim_vals.end(), obj_fn_vals);

    if (this->write_combined_sim_scores_to_stream_ || this->debug_save_iter_debug_info_)
    {
      debug_sim_val_ = std::min(debug_sim_val_,
                                      *std::min_element(sim_vals.begin(), sim_vals.end()));
    }

    // update the search distribution used for cmaes_SamplePopulation()
    cmaes_UpdateDistribution(evo, obj_fn_vals);

    if (opt_aux)
    {
      MatMxN cur_cov(tot_num_params,tot_num_params);

      for (size_type r = 0; r < tot_num_params; ++r)
      {
        for (size_type c = 0; c < r; ++c)
        {
          cur_cov(r,c) = evo->C[r][c];
          cur_cov(c,r) = evo->C[r][c];
        }
        cur_cov(r,r) = evo->C[r][r];
      }

      opt_aux->cov_mats.push_back(cur_cov);
      opt_aux->sim_vals.push_back(sim_vals);
      opt_aux->pop_params.push_back(pop_params);
    }

    this->end_of_iteration();

    ++iter;
  }

  this->after_last_iteration();

  this->update_regi_xforms(xforms_at_cur_mean());

  if (this->src_and_obj_pose_opt_vars_)
  {
    this->regi_cam_models_.assign(1, cam_model_at_cur_mean());
  }

  // clean up any memory allocated interally to CMAES
  cmaes_exit(evo);
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_opt_obj_fn_tol(const Scalar& tol)
{
  obj_fn_tol_ = tol;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::set_opt_x_tol(const Scalar& tol)
{
  x_tol_ = tol;
}

xreg::size_type xreg::Intensity2D3DRegiCMAESjustinsnake::max_num_projs_per_view_per_iter() const
{
  return this->num_projs_per_view_;
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::init_opt()
{
  // we'll init during the run call
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::debug_write_remapped_drrs()
{
  const size_type num_sim_metrics = this->sim_metrics_.size();

  for (size_type view_idx = 0; view_idx < num_sim_metrics; ++view_idx)
  {
    Path dst_path = this->debug_output_dir_path_;
    dst_path += fmt::format("drr_remap_{:03d}_{:03d}.png", this->num_obj_fn_evals_, view_idx);

    WriteITKImageRemap8bpp(debug_cur_mean_drrs_[view_idx].GetPointer(), dst_path.string());
  }
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::debug_write_raw_drrs()
{
  const size_type num_sim_metrics = this->sim_metrics_.size();

  for (size_type view_idx = 0; view_idx < num_sim_metrics; ++view_idx)
  {
    Path dst_path = this->debug_output_dir_path_;
    dst_path += fmt::format("drr_raw_{:03d}_{:03}.nii.gz", this->num_obj_fn_evals_, view_idx);

    WriteITKImageToDisk(debug_cur_mean_drrs_[view_idx].GetPointer(), dst_path.string());
  }
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::debug_write_fixed_img_edge_overlays()
{
  const size_type num_sim_metrics = this->sim_metrics_.size();

  for (size_type view_idx = 0; view_idx < num_sim_metrics; ++view_idx)
  {
    const auto img_2d_size = debug_cur_mean_drrs_[view_idx]->GetLargestPossibleRegion().GetSize();

    cv::Mat edge_img(img_2d_size[1], img_2d_size[0], cv::DataType<unsigned char>::type);
    RemapAndComputeEdges(debug_cur_mean_drrs_[view_idx].GetPointer(), &edge_img, 12, 75);

    auto fixed_u8 = ITKImageRemap8bpp(this->sim_metrics_[view_idx]->fixed_image().GetPointer());

    cv::Mat edge_overlay = OverlayEdges(ShallowCopyItkToOpenCV(fixed_u8.GetPointer()), edge_img, 1 /* 1 -> green edges */);

    Path dst_path = this->debug_output_dir_path_;
    dst_path += fmt::format("edges_{:03d}_{:03d}.png", this->num_obj_fn_evals_, view_idx);
    cv::imwrite(dst_path.string(), edge_overlay);
  }
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::write_debug()
{
  if (this->write_debug_requires_drr())
  {
    compute_drrs_at_mean();
  }

  Intensity2D3DRegi::write_debug();
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::debug_write_comb_sim_score()
{
  if (this->write_combined_sim_scores_to_stream_)
  {
    this->dout() << fmt::format("{:+20.6f}\n", debug_sim_val_);
  }

  if (this->debug_save_iter_debug_info_ && this->num_obj_fn_evals_)
  {
    this->debug_info_->sims.push_back(debug_sim_val_);
  }
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::debug_write_opt_pose_vars()
{
  const size_type nv = this->num_vols();
  const size_type num_params_per_xform = this->opt_vars_->num_params();

  const double* cur_params = get_cmaes_cur_params_buf();

  std::vector<double> cmaes_cur_params_list = {};

  auto multi_device_vec = Convert_optparams_to_snake_xformvec(cur_params);

  for(size_type vol_idx = 0; vol_idx < nv; ++vol_idx)
  {
    for(size_type idx = 0; idx < num_params_per_xform; ++idx)
    {
      cmaes_cur_params_list.push_back(static_cast<double>(multi_device_vec[vol_idx][idx]));
    }
  }

  if (this->write_opt_vars_to_stream_)
  {
    size_type global_param_idx = 0;

    for (size_type vol_idx = 0; vol_idx < nv; ++vol_idx)
    {
      for (size_type param_idx = 0; param_idx < num_params_per_xform; ++param_idx, ++global_param_idx)
      {
        this->dout() << fmt::format("{:+14.6f}, ", cur_params[global_param_idx]);
      }
      this->dout() << "| ";
    }
    this->dout() << '\n';
  }

  if (this->debug_save_iter_debug_info_ && this->num_obj_fn_evals_)
  {
    for (size_type vol_idx = 0; vol_idx < nv; ++vol_idx)
    {
      this->debug_info_->iter_vars[vol_idx].push_back(
            Eigen::Map<PtN_d>(const_cast<double*>(&cmaes_cur_params_list[vol_idx * num_params_per_xform]),
                              num_params_per_xform).cast<Scalar>());
    }
  }
}

const double* xreg::Intensity2D3DRegiCMAESjustinsnake::get_cmaes_cur_params_buf()
{
  const char* param_name = "xmean";
  //const char* param_name = "xbestever";

  return cmaes_GetPtr(evo_.get(), param_name);
}

xreg::Intensity2D3DRegiCMAESjustinsnake::ScalarList
xreg::Intensity2D3DRegiCMAESjustinsnake::get_cmaes_cur_params_list()
{
  const size_type nv = this->num_vols();
  const size_type num_params_per_xform = this->opt_vars_->num_params();
  const double* cmaes_buf = get_cmaes_cur_params_buf();

  ScalarList cmaes_cur_params_list = {};

  auto multi_device_vec = Convert_optparams_to_snake_xformvec(cmaes_buf);

  for(size_type vol_idx = 0; vol_idx < nv; ++vol_idx)
  {
    for(size_type idx = 0; idx < num_params_per_xform; ++idx)
    {
      cmaes_cur_params_list.push_back(multi_device_vec[vol_idx][idx]);
    }
  }

  return cmaes_cur_params_list;
}

xreg::FrameTransformList xreg::Intensity2D3DRegiCMAESjustinsnake::xforms_at_cur_mean()
{
  return this->opt_vec_to_frame_transforms(get_cmaes_cur_params_list());
}

xreg::CameraModel xreg::Intensity2D3DRegiCMAESjustinsnake::cam_model_at_cur_mean()
{
  xregASSERT(this->src_and_obj_pose_opt_vars_);

  return this->src_and_obj_pose_opt_vars_->cam(
                    Eigen::Map<PtN_d>(const_cast<double*>(get_cmaes_cur_params_buf()),
                                            this->src_and_obj_pose_opt_vars_->num_params()).cast<Scalar>());
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::compute_drrs_at_mean()
{
  FrameTransformList xform_list = xforms_at_cur_mean();

  CamModelList* cams = nullptr;
  CamModelList tmp_cams;

  if (this->src_and_obj_pose_opt_vars_)
  {
    cams = &tmp_cams;
    tmp_cams.assign(1, cam_model_at_cur_mean());
  }

  this->debug_compute_drrs_single_proj_per_view(xform_list, &debug_cur_mean_drrs_, true, cams);
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::OptAux::read(const H5::CommonFG& h5)
{
  xregThrow("Intens. 2D/3D Regi. CMA-ES OptAux::read() unsuported!");
}

void xreg::Intensity2D3DRegiCMAESjustinsnake::OptAux::write(H5::CommonFG* h5)
{
  H5::Group aux_g = h5->createGroup("opt-aux");

  WriteStringH5("name", "CMA-ES", &aux_g, false);

  const size_type num_its = cov_mats.size();
  WriteSingleScalarH5("num-iterations", num_its, &aux_g);

  WriteSingleScalarH5("pop-size", pop_size, &aux_g);

  xregASSERT(pop_params.size() == num_its);

  // write every population parameterization
  {
    H5::Group all_pop_params_g = aux_g.createGroup("all-pop-params");

    MatMxN single_obj_pop_per_iter(pop_size, se3_param_dim);

    for (size_type i = 0; i < num_its; ++i)
    {
      H5::Group cur_it_pop_params_g = all_pop_params_g.createGroup(fmt::format("iter-{:03d}", i));

      const auto& pop_params_cur_it = pop_params[i];

      const size_type num_objs = pop_params_cur_it.size();

      for (size_type oi = 0; oi < num_objs; ++oi)
      {
        const auto& pop_params_cur_obj = pop_params_cur_it[oi];
        xregASSERT(pop_params_cur_obj.size() == pop_size);

        for (size_type p = 0; p < pop_size; ++p)
        {
          for (size_type k = 0; k < se3_param_dim; ++k)
          {
            single_obj_pop_per_iter(p,k) = pop_params_cur_obj[p][k];
          }
        }
        WriteMatrixH5(fmt::format("obj-{:03d}-pop", oi), single_obj_pop_per_iter, &cur_it_pop_params_g);
      }
    }
  }

  // write all sim values for every iteration and every population sample
  {
    MatMxN all_sim_vals(num_its, pop_size);
    for (size_type i = 0; i < num_its; ++i)
    {
      const auto& cur_it_sim_vals = sim_vals[i];

      for (size_type p = 0; p < pop_size; ++p)
      {
        all_sim_vals(i,p) = cur_it_sim_vals[p];
      }
    }

    WriteMatrixH5("all-sim-vals", all_sim_vals, &aux_g);
  }

  // write covariances
  {
    H5::Group cov_g = aux_g.createGroup("covariances");

    for (size_type i = 0; i < num_its; ++i)
    {
      WriteMatrixH5(fmt::format("iter-{:03d}", i), cov_mats[i], &cov_g);
    }
  }

  // write number of rejections per iteration
  if (!num_rejects.empty())
  {
    xregASSERT(num_rejects.size() == num_its);
    WriteVectorH5("all-num-rejects", num_rejects, &aux_g);
  }
}
