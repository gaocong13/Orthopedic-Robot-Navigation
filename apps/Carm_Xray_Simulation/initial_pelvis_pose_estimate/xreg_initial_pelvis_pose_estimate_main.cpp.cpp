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

#include "xregProgOptUtils.h"
#include "xregFCSVUtils.h"
#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregLandmarkMapUtils.h"
#include "xregAnatCoordFrames.h"
#include "xregH5ProjDataIO.h"
#include "xregProjPreProc.h"
#include "xregCIOSFusionDICOM.h"
#include "xregPnPUtils.h"

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS   = 0;
  constexpr int kEXIT_VAL_BAD_USE   = 1;
  constexpr int kEXIT_VAL_BAD_INPUT = 2;

  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Estimate an initial pelvis pose from 2D and 3D annotations"
              "2D annotation is created from a random AP view projection image.");

  po.set_arg_usage("<3D landmark fcsv file> <2D landmark fcsv file>"
                   "<Output pelvis pose h5 file>");
  po.set_min_num_pos_args(3);

  po.add("no-ras2lps", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "no-ras2lps",
         "Do NOT convert RAS to LPS (or LPS to RAS) for the 3D landmarks; "
         "RAS to LPS conversion negates the first and second components.")
    << false;

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

  std::ostream& vout = po.vout();

  const std::string vol_landmark_path   = po.pos_args()[0];
  const std::string proj_landmark_path  = po.pos_args()[1];
  const std::string proj_img_path       = po.pos_args()[2];
  const std::string pelvis_pose_path    = po.pos_args()[3];
  const std::string proj_disk_path      = po.pos_args()[4];

  const bool ras2lps = !po.get("no-ras2lps").as_bool();

  vout << "reading 3D anatomical landmarks from FCSV file..." << std::endl;
  auto vol_anatld_fcsv = ReadFCSVFileNamePtMap(vol_landmark_path);

  if (ras2lps)
  {
    vout << "converting 3D landmarks from RAS -> LPS" << std::endl;
    ConvertRASToLPS(&vol_anatld_fcsv);
  }

  vout << "reading 2D anatomical landmarks from FCSV file..." << std::endl;
  auto proj_anatld_fcsv = ReadFCSVFileNamePtMap(proj_landmark_path);
/*
  vout << "converting RAS --> LPS..." << std::endl;
  ConvertRASToLPS(&proj_anatld_fcsv);
*/
  vout << "reading ITK image from disk..." << std::endl;
  auto proj_img = ReadITKImageFromDisk<itk::Image<float,2>>(proj_img_path);

  // LandMap3 vol_lands;
  // for( const auto& n : vol_anatld_fcsv ) {
  //   vol_lands.insert(n);
  // }
  //
  // LandMap2 proj_lands;
  // for( const auto& n : proj_anatld_fcsv ) {
  //   std::pair<std::string, Pt2> ld(n.first, Pt2(n.second[0], n.second[1]));
  //   proj_lands.insert(ld);
  // }

  ProjPreProc proj_pre_proc;
  proj_pre_proc.input_projs.resize(1);

  vout << "    creating camera model..." << std::endl;
  proj_pre_proc.input_projs[0].cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);
  proj_pre_proc.input_projs[0].img = proj_img;

  {
    vout << "updating 2D landmarks using CIOS metadata..." << std::endl;
    UpdateLandmarkMapForCIOSFusion(MakeNaiveCIOSFusionMetaDR(), &proj_anatld_fcsv);

    auto& proj_lands = proj_pre_proc.input_projs[0].landmarks;
    proj_lands.reserve(proj_anatld_fcsv.size());

    vout << "putting 2D fcsv landmarks into proj data..." << std::endl;
    for (const auto& fcsv_kv : proj_anatld_fcsv)
    {
      proj_lands.emplace(fcsv_kv.first, Pt2{ fcsv_kv.second[0], fcsv_kv.second[1] });
    }
  }

  vout << "running 2D preprocessing..." << std::endl;
  proj_pre_proc();

  auto& projs_to_regi = proj_pre_proc.output_projs;

  vout << "running PnP POSIT pose estimation..." << std::endl;
  const FrameTransform pelvis_pose_xform = PnPPOSITAndReprojCMAES(projs_to_regi[0].cam, vol_anatld_fcsv, projs_to_regi[0].landmarks);

  vout << "writing pose xform to disk..." << std::endl;
  WriteITKAffineTransform(pelvis_pose_path, pelvis_pose_xform);

  vout << "writing projection data to disk..." << std::endl;
  WriteProjDataH5ToDisk(projs_to_regi[0], proj_disk_path);

  vout << "pelvis pose:\n" << pelvis_pose_xform.matrix() << '\n';

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
