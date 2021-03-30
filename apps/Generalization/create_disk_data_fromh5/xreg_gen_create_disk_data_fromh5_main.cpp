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

  po.set_help("Projects 3D pelvis/vertebra/sacrum/femur labels and landmarks into 2D. "
              "A single-channel label map is written to disk. When the optional final "
              "positional argument is set, then a multi-channel label map is written in "
              "HDF5 format. This format accounts for the possible overlap due to the "
              "line integral nature of X-ray.");

  po.set_arg_usage("<Src H5 File> <Dst H5 File> <Spec ID> <Input vol. seg.> <3D Landmarks>"
                   "<Proj. Data File> <Pelvis Pose>  "
                   "<Output Single-Chan. 2D Seg.> [<Output Multi-Chan. 2D Seg.>]");
  po.set_min_num_pos_args(5);
  po.set_help_epilogue(fmt::format("\nIPCAI version: {}", IPCAIVersionStr()));

  po.add("pat-up", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "pat-up",
         "If necessary, rotate the output projections to have the patient oriented \"up.\" "
         "This also removes the rot-pat-up field from the output projections.")
    << false;

  po.add("preproc", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "preproc",
         "Apply standard preprocessing such as cropping and, if the \"intensity\" flag is set, "
         "log intensity conversion.")
    << false;

  po.add("ds-factor", 'd', ProgOpts::kSTORE_DOUBLE, "ds-factor",
         "Downsample the camera (2D image) dimensions by this factor, this is done before "
         "projection. e.g. 0.5 --> shrink ny half, 1.0 --> no downsampling.")
    << 1.0;

  po.add("proj-idx", 'p', ProgOpts::kSTORE_UINT32, "proj-idx",
         "Index of the projection to project to (the projection data file may store several projections)")
    << ProgOpts::uint32(0);

  po.add("intensity", 'i', ProgOpts::kSTORE_STRING, "intensity",
         "Save a copy of the intensity projection data at this path, after cropping, downsampling, etc.")
    << "";

  po.add("no-ras2lps", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "no-ras2lps",
         "Do NOT convert RAS to LPS (or LPS to RAS) for the 3D landmarks; "
         "RAS to LPS conversion negates the first and second components.")
    << false;
  po.add("vol-rot", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_DOUBLE, "vol-rot",
         "Rotation Magnitude for randomizing projection geometry w.r.t. pelvis center reference frame")
    << 75.0;
  po.add("vol-transLR", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_DOUBLE, "vol-transLR",
         "Left & Right Translational Magnitude for randomizing projection geometry w.r.t. pelvis center reference frame")
    << 50.0;
  po.add("vol-transUD", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_DOUBLE, "vol-transUD",
         "Up & Down Translational Magnitude for randomizing projection geometry w.r.t. pelvis center reference frame")
    << 20.0;
  po.add("vol-transAP", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_DOUBLE, "vol-transAP",
         "Depth (AP view) Translational Magnitude for randomizing projection geometry w.r.t. pelvis center reference frame")
    << 100.0;
  po.add("num-projs", 'n', ProgOpts::kSTORE_UINT32, "num-projs",
         "Number of projections to project")
    << ProgOpts::uint32(1);
  po.add("write-projs-h5", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_TRUE, "write-projs-h5",
         "Write projs to dst h5 file, default is False")
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

  const std::string src_h5_path             = po.pos_args()[0];
  const std::string dst_h5_path             = po.pos_args()[1];
  const std::string vol_3d_path             = po.pos_args()[2];
  const std::string seg_3d_path             = po.pos_args()[3];
  const std::string fcsv_3d_path            = po.pos_args()[4];
  const std::string output_path             = po.pos_args()[5];

  std::vector<std::string> spec_ID_list(po.pos_args().begin() + 6, po.pos_args().end());
  // const std::string multi_chan_2d_seg_path = (po.pos_args().size() > 7) ? po.pos_args()[7] : std::string();

  const double ds_factor = po.get("ds-factor");

  const bool do_ds = std::abs(ds_factor - 1.0) > 0.001;

  const bool ras2lps = !po.get("no-ras2lps").as_bool();

  const size_type proj_idx = po.get("proj-idx").as_uint32();

  const std::string dst_intens_proj_path = po.get("intensity");

  const bool save_intens_proj = !dst_intens_proj_path.empty();

  const bool apply_preproc = po.get("preproc");

  const bool rot_pat_up = po.get("pat-up");

  const double vol_rot_deg = po.get("vol-rot");

  const double vol_transLR = po.get("vol-transLR");

  const double vol_transUD = po.get("vol-transUD");

  const double vol_transAP = po.get("vol-transAP");

  const size_type num_projs = po.get("num-projs").as_uint32();

  const bool write_projs_h5 = po.get("write-projs-h5");

  const size_type num_specs = spec_ID_list.size();

  vout << "opening output file for reading..." << std::endl;
  H5::H5File src_h5(src_h5_path, H5F_ACC_RDWR);

  vout << "opening output file for writing..." << std::endl;
  H5::H5File dst_h5(dst_h5_path, H5F_ACC_TRUNC);

  const auto default_cam = NaiveCamModelFromCIOSFusion(MakeNaiveCIOSFusionMetaDR(), true);

  // map world --> camera 3D
  const FrameTransform extrins = default_cam.extrins;

  // project camera 3D --> image plane 2D pixels
  const Mat3x3 intrins = default_cam.intrins;

  // Every projection should have the same extrinsics/intrinsics for this dataset
  vout << "writing camera params..." << std::endl;
  {
    H5::Group cam_g = dst_h5.createGroup("proj-params");

    WriteSingleScalarH5("num-rows", default_cam.num_det_rows, &cam_g);
    WriteSingleScalarH5("num-cols", default_cam.num_det_cols, &cam_g);

    WriteSingleScalarH5("pixel-row-spacing", default_cam.det_row_spacing, &cam_g);
    WriteSingleScalarH5("pixel-col-spacing", default_cam.det_col_spacing, &cam_g);

    WriteMatrixH5("intrinsic", intrins, &cam_g);
    WriteAffineTransform4x4("extrinsic", extrins, &cam_g);
  }

  const std::vector<std::string> land_names = { "FH-l",   "FH-r",
                                                "GSN-l",  "GSN-r",
                                                "IOF-l",  "IOF-r",
                                                "MOF-l",  "MOF-r",
                                                "SPS-l",  "SPS-r",
                                                "IPS-l",  "IPS-r",
                                                "ASIS-l", "ASIS-r" };

  const size_type proj_num_lands = land_names.size();

  vout << "writing landmark names..." << std::endl;
  {
    H5::Group dst_land_names_g = dst_h5.createGroup("land-names");

    WriteSingleScalarH5("num-lands", proj_num_lands, &dst_land_names_g);

    for (size_type land_idx = 0; land_idx < proj_num_lands; ++land_idx)
    {
      WriteStringH5(fmt::format("land-{:02d}", land_idx), land_names[land_idx], &dst_land_names_g);
    }
  }

  std::ofstream log_txt;
  log_txt.open(output_path + "/log.txt", std::ios::app);

  for(size_type spec_idx = 0; spec_idx < num_specs; ++spec_idx)
  {
    bool need_to_init_ds = true;

    const std::string spec_ID = spec_ID_list[spec_idx];
    vout << "reading input CT volume..." << std::endl;
    const std::string vol_3d_file = vol_3d_path + '/' + spec_ID + ".nrrd";
    auto vol_hu = ReadITKImageFromDisk<RayCaster::Vol>(vol_3d_file);

    vout << "converting HU --> Lin. Att." << std::endl;
    auto vol_intens = HUToLinAtt(vol_hu.GetPointer(), -130);

    vout << "reading label map..." << std::endl;
    const std::string seg_3d_file = seg_3d_path + '/' + spec_ID + "_Seg.nrrd";
    auto ct_labels = ReadITKImageFromDisk<itk::Image<unsigned char,3>>(seg_3d_file);

    //////////////////////////////////////////////////////////////////////////////
    // Get the landmarks

    vout << "reading 3D landmarks..." << std::endl;
    const std::string lands_3d_file = fcsv_3d_path + '/' + spec_ID + "_landmark.fcsv";
    LandMap3 lands_3d = ReadFCSVFileNamePtMap(lands_3d_file);

    if (ras2lps)
    {
      vout << "converting landmarks from RAS -> LPS" << std::endl;
      ConvertRASToLPS(&lands_3d);
    }

    vout << "3D Landmarks:\n";
    PrintLandmarkMap(lands_3d, vout);

    vout << "creating spec: " << spec_ID << " in dst_h5_file" << std::endl;
    H5::Group dst_spec_g = dst_h5.createGroup(spec_ID);

    H5::Group gt_poses_g = dst_spec_g.createGroup("gt-poses");

    vout << "reading spec: " << spec_ID << " in src_h5_file" << std::endl;
    H5::Group src_spec_g = src_h5.openGroup(spec_ID);

    H5::Group src_gt_poses_g = src_spec_g.openGroup("gt-poses");

    H5::DataSet proj_ds;
    H5::DataSet seg_ds;
    H5::DataSet lands_ds;

    using LandMatRef = Eigen::Map<Eigen::Matrix<CoordScalar,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor|Eigen::DontAlign>>;

    std::vector<CoordScalar> land_buf(proj_num_lands * 2);
    LandMatRef lands_buf_mat_ref(&land_buf[0], 2, proj_num_lands);

    for(size_type rand_idx = 0; rand_idx < num_projs; ++rand_idx)
    {
      std::cout << "running spec " << spec_ID << " rand idx " << rand_idx << std::endl;
      auto pelvis_pose = ReadAffineTransform4x4H5(fmt::format("{:04d}", rand_idx), src_gt_poses_g);;

      WriteAffineTransform4x4(fmt::format("{:04d}", rand_idx), pelvis_pose, &gt_poses_g);

      vout << "setting up ray caster..." << std::endl;
      auto rc = LineIntRayCasterFromProgOpts(po);

      rc->set_camera_models(RayCaster::CameraModelList(1,default_cam));
      rc->set_num_projs(1);
      rc->set_volume(vol_intens);

      vout << "  allocating resources..." << std::endl;
      rc->allocate_resources();

      rc->distribute_xform_among_cam_models(pelvis_pose);

      vout << "  ray casting..." << std::endl;
      rc->compute();

      ProjDataF32 pd;

      const size_type num_photons = 2000;

      pd.cam = default_cam;
      // pd.img = SamplePoissonProjFromAttProj(rc->proj(0).GetPointer(), num_photons);
      pd.img = rc->proj(0).GetPointer();

      if (apply_preproc)
      {
        vout << "projection preprocessing..." << std::endl;

        ProjPreProc preproc;
        preproc.set_debug_output_stream(vout, verbose);

        preproc.input_projs = { pd };

        preproc();

        pd = preproc.output_projs[0];
      }

      if (do_ds)
      {
        vout << "downsampling input projection data..." << std::endl;
        pd = DownsampleProjData(pd, ds_factor);
      }

      const auto& cam = pd.cam;

      const FrameTransform left_femur_pose = pelvis_pose;
      const FrameTransform right_femur_pose = pelvis_pose;
      auto proj_labels = ProjectHipLabels(ct_labels, cam, pelvis_pose, left_femur_pose, right_femur_pose,
                                          ProgOptsDepthRayCasterFactory(po), verbose, vout);

      vout << "creating proj. data. struct for the single-channel segmentation..."  << std::endl;
      ProjDataU8 seg_pd;

      {
        const FrameTransform vol_to_cam = pelvis_pose.inverse();

        seg_pd.cam = cam;
        seg_pd.img = std::get<0>(proj_labels);

        for (const auto& lkv : lands_3d)
        {
          seg_pd.landmarks.emplace(lkv.first, cam.phys_pt_to_ind_pt(vol_to_cam * lkv.second).head(2));
        }
      }

      if (rot_pat_up && pd.rot_to_pat_up && (*pd.rot_to_pat_up != ProjDataRotToPatUp::kZERO))
      {
        xregASSERT(*pd.rot_to_pat_up == ProjDataRotToPatUp::kONE_EIGHTY);

        vout << "rotating projections for patient up..." << std::endl;

        ModifyForPatUp(seg_pd.img.GetPointer(), *pd.rot_to_pat_up);

        for (auto& lkv : seg_pd.landmarks)
        {
          lkv.second[0] = static_cast<CoordScalar>(cam.num_det_cols - 1) - lkv.second[0];
          lkv.second[1] = static_cast<CoordScalar>(cam.num_det_rows - 1) - lkv.second[1];
        }

        if (save_intens_proj)
        {
          ModifyForPatUp(pd.img.GetPointer(), *pd.rot_to_pat_up);
          pd.rot_to_pat_up.reset();
        }
      }
      else
      {
        seg_pd.rot_to_pat_up = pd.rot_to_pat_up;
      }

      if (save_intens_proj)
      {
        pd.landmarks = seg_pd.landmarks;

        vout << "writing processed intensity proj. data to disk..." << std::endl;
        WriteProjDataH5ToDisk(pd, dst_intens_proj_path);
      }

      const std::string zfill_idx = fmt::format("{:05d}", rand_idx);
      {
        vout << "      writing png seg image object to disk..." << std::endl;
        const std::string seg_png_path = output_path + "/seg/" + zfill_idx + ".png";
        WriteITKImageToDisk(seg_pd.img.GetPointer(), seg_png_path);

        vout << "      writing tiff proj image object to disk..." << std::endl;
        const std::string proj_tiff_path = output_path + "/proj/" + zfill_idx + ".tiff";
        WriteITKImageToDisk(pd.img.GetPointer(), proj_tiff_path);

        vout << "      writing landmark data to disk..." << std::endl;
        std::vector<CoordScalar> lands_dim1(proj_num_lands);
        std::vector<CoordScalar> lands_dim2(proj_num_lands);

        for (size_type land_idx = 0; land_idx < proj_num_lands; ++land_idx)
        {
          const auto cur_land_it = seg_pd.landmarks.find(land_names[land_idx]);
          xregASSERT(cur_land_it != seg_pd.landmarks.end());

          lands_dim1[land_idx] = cur_land_it->second(0);
          lands_dim2[land_idx] = cur_land_it->second(1);
        }

        H5::H5File h5_lands(output_path + "/lands/" + zfill_idx + ".h5", H5F_ACC_TRUNC);
        WriteVectorH5("dim1", lands_dim1, &h5_lands);
        WriteVectorH5("dim2", lands_dim2, &h5_lands);
        h5_lands.flush(H5F_SCOPE_GLOBAL);
        h5_lands.close();
      }

      {
        vout << "Creating log txt..." << std::endl;
        log_txt << "zfillID:" <<  zfill_idx << ' '
                << "specID:" << spec_ID << ' '
                << "randID:" << rand_idx << '\n';
      }

      if(write_projs_h5)
      {
        vout << "writing png proj image object to disk..." << std::endl;
        const std::string proj_png_path = output_path + "/" + spec_ID + "_proj_" + zfill_idx + ".png";
        WriteITKImageRemap8bpp(pd.img.GetPointer(), proj_png_path);

        const std::string proj_disk_path = output_path + "/" + spec_ID + "_proj_" + zfill_idx + ".h5";
        WriteProjDataH5ToDisk(pd, proj_disk_path);
      }

      const size_type proj_num_cols = pd.cam.num_det_cols;
      const size_type proj_num_rows = pd.cam.num_det_rows;

      if (need_to_init_ds)
      {
        // Create projs & segs dataset
        {
          const std::array<hsize_t,3> proj_ds_dims = { num_projs, proj_num_rows, proj_num_cols };

          H5::DataSpace data_space(proj_ds_dims.size(), proj_ds_dims.data());

          // chunks for compression
          H5::DSetCreatPropList props;
          props.copy(H5::DSetCreatPropList::DEFAULT);

          // chunk of 1 image
          std::array<hsize_t,3> chunk_dims = { 1, proj_num_rows, proj_num_cols };
          props.setChunk(chunk_dims.size(), chunk_dims.data());
          props.setDeflate(9);

          if(write_projs_h5)
          {
            vout << "        creating projs dataset..." << std::endl;
            proj_ds = dst_spec_g.createDataSet("projs", LookupH5DataType<RayCaster::PixelScalar2D>(), data_space, props);
          }

          vout << "        creating segs dataset..." << std::endl;
          seg_ds = dst_spec_g.createDataSet("segs", LookupH5DataType<unsigned char>(), data_space, props);
        }

        // Create lands dataset
        {
          const std::array<hsize_t,3> lands_ds_dims = { num_projs, 2, proj_num_lands };

          H5::DataSpace data_space(lands_ds_dims.size(), lands_ds_dims.data());

          // chunks for compression
          H5::DSetCreatPropList props;
          props.copy(H5::DSetCreatPropList::DEFAULT);

          // chunk the entire array
          std::array<hsize_t,3> chunk_dims = { num_projs, 2, proj_num_lands };
          props.setChunk(chunk_dims.size(), chunk_dims.data());
          props.setDeflate(9);

          vout << "      creating lands dataset..." << std::endl;
          lands_ds = dst_spec_g.createDataSet("lands", LookupH5DataType<CoordScalar>(), data_space, props);
        }

        need_to_init_ds = false;
      }

      {
        const std::array<hsize_t,3> m_dims = { 1, proj_num_rows, proj_num_cols };
        H5::DataSpace ds_m(m_dims.size(), m_dims.data());

        const std::array<hsize_t,3> f_start = { rand_idx, 0, 0 };

        if(write_projs_h5)
        {
          vout << "      writing projection data..." << std::endl;

          H5::DataSpace ds_f = proj_ds.getSpace();
          ds_f.selectHyperslab(H5S_SELECT_SET, m_dims.data(), f_start.data());

          proj_ds.write(pd.img->GetBufferPointer(), LookupH5DataType<RayCaster::PixelScalar2D>(), ds_m, ds_f);
        }

        {
          vout << "      writing segmentation data..." << std::endl;

          H5::DataSpace ds_f = seg_ds.getSpace();
          ds_f.selectHyperslab(H5S_SELECT_SET, m_dims.data(), f_start.data());

          seg_ds.write(seg_pd.img->GetBufferPointer(), LookupH5DataType<unsigned char>(), ds_m, ds_f);
        }

      }

      {
        vout << "      writing landmark data..." << std::endl;

        for (size_type land_idx = 0; land_idx < proj_num_lands; ++land_idx)
        {
          const auto cur_land_it = seg_pd.landmarks.find(land_names[land_idx]);
          xregASSERT(cur_land_it != seg_pd.landmarks.end());

          lands_buf_mat_ref(0,land_idx) = cur_land_it->second(0);
          lands_buf_mat_ref(1,land_idx) = cur_land_it->second(1);
        }

        const std::array<hsize_t,3> m_dims = { 1, 2, proj_num_lands };
        H5::DataSpace ds_m(m_dims.size(), m_dims.data());

        const std::array<hsize_t,3> f_start = { rand_idx, 0, 0 };

        H5::DataSpace ds_f = lands_ds.getSpace();
        ds_f.selectHyperslab(H5S_SELECT_SET, m_dims.data(), f_start.data());

        // recall that lands_buf_mat_ref uses land_buf
        lands_ds.write(&land_buf[0], LookupH5DataType<CoordScalar>(), ds_m, ds_f);
      }
    }
  }

  dst_h5.flush(H5F_SCOPE_GLOBAL);
  dst_h5.close();

  src_h5.flush(H5F_SCOPE_GLOBAL);
  src_h5.close();

  log_txt.close();
  vout << "exiting..." << std::endl;

  return kEXIT_VAL_SUCCESS;
}
