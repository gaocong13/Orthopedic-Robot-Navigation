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

#include <fmt/format.h>

#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregLandmarkMapUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregCIOSFusionDICOM.h"
#include "xregSampleUtils.h"
#include "xregSampleUniformUnitVecs.h"
#include "xregRotUtils.h"
#include "xregRigidUtils.h"
#include "xregH5ProjDataIO.h"
#include "xregOpenCVUtils.h"
#include "xregHDF5.h"
#include "xregProjPreProc.h"
#include "xregRayCastProgOpts.h"
#include "xregRayCastInterface.h"
#include "xregHUToLinAtt.h"
#include "xregImageAddPoissonNoise.h"

#include "IPCAICommon.h"

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;

  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Generate IPCAI data groundtruth h5 file by projecting each specimen CT"
              "to lands and segs using groundtruth pelvis pose.");

  po.set_arg_usage("<IPCAI full res H5 File> <output path>");
  po.set_min_num_pos_args(2);
  po.set_help_epilogue(fmt::format("\nIPCAI version: {}", IPCAIVersionStr()));

  po.add("no-ras2lps", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "no-ras2lps",
         "Do NOT convert RAS to LPS (or LPS to RAS) for the 3D landmarks; "
         "RAS to LPS conversion negates the first and second components.")
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

  const std::string ipcai_ful_res_h5_path   = po.pos_args()[0];
  const std::string output_path             = po.pos_args()[1];

  const bool ras2lps = !po.get("no-ras2lps").as_bool();

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  std::vector<std::string> spec_ID_list = {"17-1882", "18-1109", "18-0725", "18-2799", "18-2800", "17-1905"};
  std::vector<size_type> spec_projs_num_list = {111, 104, 24, 48, 55, 24};
  const size_type num_specs = 6;

  H5::H5File h5(ipcai_ful_res_h5_path, H5F_ACC_RDWR);

  for(size_type spec_idx = 0; spec_idx < num_specs; ++spec_idx)
  {
    const std::string spec_ID = spec_ID_list[spec_idx];
    const size_type spec_num = spec_projs_num_list[spec_idx];

    H5::Group ipcai_spec_ID_g = h5.openGroup(spec_ID);

    H5::Group ipcai_spec_ID_projections_g = ipcai_spec_ID_g.openGroup("projections");

    vout << "reading label map..." << std::endl;
    H5::Group ipcai_spec_ID_seg_g = ipcai_spec_ID_g.openGroup("vol-seg/image");
    auto ct_labels = ReadITKImageH5UChar3D(ipcai_spec_ID_seg_g);

    // WriteITKImageToDisk(ct_labels.GetPointer(), output_path + "/" + spec_ID + "_Seg.nii.gz");

    ProjDataF32List pjs;
    pjs.resize(spec_num);

    FrameTransformList pel_xform_list;

    for(size_type proj_idx = 0; proj_idx < spec_num; ++proj_idx)
    {
      std::cout << "running spec: " << spec_ID << " proj idx: " << proj_idx << std::endl;

      const std::string proj_idx_fmt = fmt::format("{:03d}", proj_idx);

      H5::Group ipcai_proj_g = ipcai_spec_ID_projections_g.openGroup(proj_idx_fmt);

      H5::Group ipcai_gt_poses_g = ipcai_proj_g.openGroup("gt-poses");

      const FrameTransform pelvis_pose = ReadAffineTransform4x4H5("cam-to-pelvis-vol", ipcai_gt_poses_g);

      pel_xform_list.push_back(pelvis_pose);

      // WriteITKAffineTransform(output_path + "/pel_pose" + spec_ID + "_" + proj_idx_fmt + ".h5", pelvis_pose);

      H5::Group ipcai_proj_image_g = ipcai_proj_g.openGroup("image");

      auto real_proj = ReadITKImageH5Float2D(ipcai_proj_image_g);

      pjs[proj_idx].img = real_proj.GetPointer();
    }

    std::vector<CameraModel> orig_cams;
    for (auto& pd : pjs)
    {
      orig_cams.push_back(default_cam);
    }

    // Using device regi as fiducial
    auto cams_devicefid = CreateCameraWorldUsingFiducial(orig_cams, pel_xform_list);

    for(size_type cam_idx = 0; cam_idx < spec_num; ++cam_idx)
    {
      pjs[cam_idx].cam = cams_devicefid[cam_idx];
    }

    if(false)
    {
      ProjPreProc preproc;
      preproc.set_debug_output_stream(vout, verbose);

      preproc.input_projs = { pjs };

      preproc();

      auto pjs = preproc.output_projs;

      pjs = DownsampleProjData(pjs, 0.25);
    }

    const std::string proj_data_h5_path = output_path + "/projs/proj" + spec_ID + ".h5";
    H5::H5File h5(proj_data_h5_path, H5F_ACC_TRUNC);

    WriteProjDataH5(pjs, &h5);
  }

  WriteITKAffineTransform(output_path + "/pel_identity.h5", FrameTransform::Identity());

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
