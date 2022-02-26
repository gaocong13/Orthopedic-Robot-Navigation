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
 // STD
 #include <iostream>
 #include <vector>
 
#include <fmt/format.h>

#include "xregProgOptUtils.h"
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
#include "xregImageIntensLogTrans.h"

int main(int argc, char* argv[])
{
  using namespace xreg;

  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;

  // First, set up the program options

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help(" Convert DCM image to tiff by performing downsampling, log remapping. ");

  po.set_arg_usage("< Image Name (ID) > < Image Load Path > < Image Write Path >");
  po.set_min_num_pos_args(3);

  po.add("logxform", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "logxform",
         "log intensity conversion.")
    << false;

  po.add("preproc", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "preproc",
         "Apply standard preprocessing such as cropping and, if the \"intensity\" flag is set, "
         "log intensity conversion.")
    << false;

  po.add("ds-factor", 'd', ProgOpts::kSTORE_DOUBLE, "ds-factor",
         "Downsample the camera (2D image) dimensions by this factor, this is done before "
         "projection. e.g. 0.5 --> shrink ny half, 1.0 --> no downsampling.")
    << 1.0;

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

  const std::string image_ID     = po.pos_args()[0];
  const std::string image_path   = po.pos_args()[1];
  const std::string write_path   = po.pos_args()[2];

  const bool do_logxform = po.get("logxform");
  const double ds_factor = po.get("ds-factor");
  const bool do_ds = std::abs(ds_factor - 1.0) > 0.001;
  const bool apply_preproc = po.get("preproc");

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  ProjDataF32 pd;

  const std::string img_file = image_path + "/" + image_ID;
  vout << "reading image..." << img_file << "\n";
  pd.cam = default_cam;
  pd.img = ReadITKImageFromDisk<itk::Image<float,2>>(img_file);

  const auto& cam = pd.cam;

  if (apply_preproc)
  {
    vout << "projection preprocessing..." << std::endl;

    ProjPreProc preproc;
    preproc.set_debug_output_stream(vout, verbose);

    preproc.input_projs = { pd };

    preproc();

    pd = preproc.output_projs[0];
  }

  if (do_logxform)
  {
    vout << "Log transforming..." << std::endl;

    auto log_xform = ImageIntensLogTransFilter::New();
    log_xform->SetInput(pd.img);
    log_xform->SetUseMaxIntensityAsI0(true);
    log_xform->Update();

    pd.img = log_xform->GetOutput();
  }

  if (do_ds)
  {
    vout << "downsampling input projection data..." << std::endl;
    pd = DownsampleProjData(pd, ds_factor);
  }

  vout << "writing img to disk..." << std::endl;
  const std::string proj_tiff_path = write_path + "/" + image_ID + ".tiff";
  WriteITKImageToDisk(pd.img.GetPointer(), proj_tiff_path);

  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
