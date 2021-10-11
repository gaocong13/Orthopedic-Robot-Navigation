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

#include <itkBinaryThresholdImageFilter.h>
#include <itkFlipImageFilter.h>

// xreg
#include "xregStringUtils.h"
#include "xregMeshIO.h"
#include "xregProgOptUtils.h"
#include "xregVTKMeshUtils.h"
#include "xregITKIOUtils.h"

using namespace xreg;

int main(int argc, char* argv[])
{
  const int kEXIT_VAL_SUCCESS = 0;
  const int kEXIT_VAL_BAD_USE = 1;

  ProgOpts po;

  xregPROG_OPTS_SET_COMPILE_DATE(po);

  po.set_help("Transform a mesh object using a rigid transformtation. Then save "
              "the transformed mesh file to disk.");
  po.set_arg_usage("input mesh> <rigid transformation> <output mesh> ");

  po.set_min_num_pos_args(3);

  po.add("invert-xform", 'i', ProgOpts::kSTORE_TRUE, "invert-xform",
         "Invert the rigid transformation.")
     << false;
  po.add("ascii", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "ascii",
        "Write into an ASCII compatible format when possible/supported.")
     << false;
  po.add("swap-lps-ras", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE,
        "swap-lps-ras",
        "Converts mesh vertices from LPS to RAS (or vice-versa)")
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

  const bool prefer_ascii = po.get("ascii");


  const std::string input_mesh_path = po.pos_args()[0];
  const std::string xform_path = po.pos_args()[1];
  const std::string output_mesh_path = po.pos_args()[2];

  const bool invert_xform = po.get("invert-xform");
  const bool swap_lps_ras = po.get("swap-lps-ras");

  vout << "reading mesh..." << std::endl;
  auto mesh = ReadMeshFromDisk(input_mesh_path);

  vout << "reading transformation..." << std::endl;
  auto xform = ReadITKAffineTransformFromFile(xform_path);

  if(invert_xform)
  {
    vout << "   transformation is inverted!" << std::endl;
    xform = xform.inverse();
  }

  if (swap_lps_ras)
  {
    vout << "LPS -> RAS..." << std::endl;
    FrameTransform lps2ras = FrameTransform::Identity();
    lps2ras.matrix()(0,0) = -1;
    lps2ras.matrix()(1,1) = -1;
    mesh.transform(lps2ras);
  }
  
  vout << "transforming mesh..." << std::endl;
  mesh.transform(xform);

  // Write mesh to disk
  vout << "writing mesh to disk..." << std::endl;
  WriteMeshToDisk(mesh, output_mesh_path, prefer_ascii);

  return kEXIT_VAL_SUCCESS;
}
