//  Created by Cong Gao on Nov.05 2021.
//  Copyright © 2021 Cong Gao. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include "itkCenteredAffineTransform.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkTileImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkAddImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkWindowedSincInterpolateImageFunction.h"
#include "itkImage.h"
#include "math.h"
//#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "omp.h"
#include "assert.h"

#include "itkAffineTransform.h"
#include <itkEuler3DTransform.h>

#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregHUToLinAtt.h"

#include "xregAssert.h"

// #include "spline.h"

const unsigned int Dimension = 3;
constexpr unsigned int Radius = 3;
const float PI = 3.14159;

namespace xreg
{
  typedef float  InputPixelType;
  typedef float  OutputPixelType;
  typedef float  ScalarType;

  using Vol         = itk::Image<InputPixelType, 3>;
  using VolPtr      = typename Vol::Pointer;

  typedef itk::ImageFileReader< Vol  >  ReaderType;
  typedef itk::ImageFileWriter< Vol >  WriterType;
  typedef itk::AddImageFilter< Vol, Vol > AddFilterType;
  typedef itk::ResampleImageFilter< Vol, Vol, ScalarType, ScalarType > ResampleFilterType;

  using InterpolatorType = itk::WindowedSincInterpolateImageFunction< Vol, Radius >;
  using TransformType = itk::AffineTransform< ScalarType, Dimension >;
  using MatrixType = itk::Matrix<ScalarType, Dimension + 1, Dimension + 1>;
  using EulerTransformType = itk::Euler3DTransform< ScalarType >;

  class BuildSnakeModel
  {
    public:
      BuildSnakeModel(const std::string meta_data_path,
                             const float rot_angX_rand,
                             const float rot_angY_rand,
                             const float rot_angZ_rand,
                             std::ostream& vout);
      ~BuildSnakeModel();

      VolPtr run();

      size_type num_vols = 28;
      std::ostream& vout_;

    private:
      void LoadData();
      void AddVolumes(AddFilterType::Pointer addFilter, VolPtr inputImage1, VolPtr inputImage2);
      void TransformVolume(ResampleFilterType::Pointer resample, VolPtr inputImage, EulerTransformType::Pointer transform);
      void SetEulerTransform(EulerTransformType::Pointer vol_xform,
                                              ScalarType rot_angX,
                                              ScalarType rot_angY,
                                              ScalarType rot_angZ,
                                              ScalarType transX,
                                              ScalarType transY,
                                              ScalarType transZ,
                                              Pt3 vol_rot_cen);
      FrameTransform ComputeAffineTransform(EulerTransformType::Pointer vert_xform,
                                                                     ScalarType transX,
                                                                     ScalarType transY,
                                                                     ScalarType transZ);
      std::vector<VolPtr> _snake_vol_list;
      std::vector<Pt3> _notch_rot_cen_list;
      LandMap3 _notch_rot_cen_fcsv;

      float _rot_angX_rand;
      float _rot_angY_rand;
      float _rot_angZ_rand;
      std::string _meta_data_path;
  };

  BuildSnakeModel::BuildSnakeModel(const std::string meta_data_path,
                                         const float rot_angX_rand,
                                         const float rot_angY_rand,
                                         const float rot_angZ_rand,
                                         std::ostream& vout):vout_(vout)
  {
    _rot_angX_rand = rot_angX_rand;
    _rot_angY_rand = rot_angY_rand;
    _rot_angZ_rand = rot_angZ_rand;
    _meta_data_path = meta_data_path;
  }

  BuildSnakeModel::~BuildSnakeModel(){}

  // BuildSnakeModel::~BuildSnakeModel();

  void BuildSnakeModel::SetEulerTransform(EulerTransformType::Pointer vol_xform,
                                                               ScalarType rot_angX,
                                                               ScalarType rot_angY,
                                                               ScalarType rot_angZ,
                                                               ScalarType transX,
                                                               ScalarType transY,
                                                               ScalarType transZ,
                                                               Pt3 vol_rot_cen)
  {
    EulerTransformType::ParametersType vol_parameters(6);
    vol_parameters[0] = rot_angX * kDEG2RAD;
    vol_parameters[1] = rot_angY * kDEG2RAD;
    vol_parameters[2] = rot_angZ * kDEG2RAD;
    vol_parameters[3] = transX;
    vol_parameters[4] = transY;
    vol_parameters[5] = transZ;
    vol_xform->SetParameters(vol_parameters);

    EulerTransformType::FixedParametersType vol_fixedparameters(3);
    vol_fixedparameters[0] = vol_rot_cen[0];
    vol_fixedparameters[1] = vol_rot_cen[1];
    vol_fixedparameters[2] = vol_rot_cen[2];
    vol_xform->SetFixedParameters(vol_fixedparameters);
  }

  FrameTransform BuildSnakeModel::ComputeAffineTransform(EulerTransformType::Pointer vol_xform,
                                                                 ScalarType transX,
                                                                 ScalarType transY,
                                                                 ScalarType transZ)
  {
    FrameTransform vol_xform4x4;
    auto rot_mat = vol_xform->GetMatrix();
    for(size_type i = 0; i < 3; ++i)
      for(size_type j = 0; j < 3; ++j)
      {
        vol_xform4x4(i, j) = rot_mat[i][j];
      }
    vol_xform4x4(0, 3) = transX;
    vol_xform4x4(1, 3) = transY;
    vol_xform4x4(2, 3) = transZ;

    return vol_xform4x4;
  }

  void BuildSnakeModel::AddVolumes(AddFilterType::Pointer addFilter,
                                                      VolPtr inputImage1,
                                                      VolPtr inputImage2)
  {
      addFilter->SetInput( 0, inputImage1 );
      addFilter->SetInput( 1, inputImage2 );
      addFilter->Update();
  }

  void BuildSnakeModel::LoadData()
  {
    for(size_type vol_idx = 0; vol_idx < num_vols; vol_idx++)
    {
      vout_ << "   [LoadData] - Loading snake volume..." << fmt::format("{:03d}", vol_idx) << std::endl;
      const std::string cur_vol_path = _meta_data_path + "/" + fmt::format("{:03d}.nrrd", vol_idx);
      auto cur_vol = ReadITKImageFromDisk<Vol>(cur_vol_path);
      _snake_vol_list.push_back(cur_vol);
    }

    vout_ << "   [LoadData] - Loading snake notch rotation centers from FCSV file..." << std::endl;
    const std::string notch_rot_cen_fcsv_path = _meta_data_path + "/notch_rot_cen.fcsv";
    _notch_rot_cen_fcsv = ReadFCSVFileNamePtMap(notch_rot_cen_fcsv_path);
    ConvertRASToLPS(&_notch_rot_cen_fcsv);

    for(size_type vol_idx = 1; vol_idx < num_vols; ++vol_idx)
    {
      auto vol_idx_name = fmt::format("{:03d}", vol_idx);

      auto fcsv_finder = _notch_rot_cen_fcsv.find(vol_idx_name);
      if(fcsv_finder != _notch_rot_cen_fcsv.end())
      {
        Pt3 notch_rot_cen_pt = fcsv_finder->second;
        _notch_rot_cen_list.push_back(notch_rot_cen_pt);
      }
      else
      {
        std::cerr << "ERROR: NOT FOUND notch rotation center: " << vol_idx << std::endl;
      }
    }
  }

  void BuildSnakeModel::TransformVolume(ResampleFilterType::Pointer resample_filter,
                                                             VolPtr inputImage,
                                        EulerTransformType::Pointer transform)
  {
    const Vol::SizeType & size = inputImage->GetLargestPossibleRegion().GetSize();
    resample_filter->SetInput(inputImage);
    resample_filter->SetReferenceImage(inputImage);
    resample_filter->UseReferenceImageOn();
    resample_filter->SetSize(size);
    resample_filter->SetDefaultPixelValue(0);

    // InterpolatorType::Pointer interpolator = InterpolatorType::New();
    // resample->SetInterpolator(interpolator);

    // get transform parameters from MatrixType
    /*
    TransformType::Pointer transform = TransformType::New();
    TransformType::ParametersType parameters(Dimension * Dimension + Dimension);
    for (unsigned int i = 0; i < Dimension; i++)
    {
      for (unsigned int j = 0; j < Dimension; j++)
      {
        parameters[i * Dimension + j] = matrix[i][j];
      }
    }
    for (unsigned int i = 0; i < Dimension; i++)
    {
      parameters[i + Dimension * Dimension] = matrix[i][Dimension];
    }
    transform->SetParameters(parameters);
    */
    resample_filter->SetTransform(transform);
  }

  std::ifstream& GotoLine(std::ifstream& file, unsigned int num){
      file.seekg(std::ios::beg);
      for(int i=0; i < num - 1; ++i){
          file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
      }
      return file;
  }

  VolPtr BuildSnakeModel::run()
  {
    LoadData();

    std::vector<VolPtr> vol_resample_list;
    ScalarType rot_angX = 0.;
    ScalarType rot_angY = 0.;
    ScalarType rot_angZ = 0.;
    ScalarType transX = 0.;
    ScalarType transY = 0.;
    ScalarType transZ = 0.;

    AddFilterType::Pointer addFilter = AddFilterType::New();
    VolPtr trans_vol = Vol::New();

    for(size_type vol_idx = 0; vol_idx < num_vols; ++vol_idx)
    {
      ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
      EulerTransformType::Pointer eul_vol_xform = EulerTransformType::New();

      rot_angX += _rot_angX_rand;
      rot_angY += _rot_angY_rand;
      rot_angZ += _rot_angZ_rand;

      ScalarType dist_rot_cen = (_notch_rot_cen_list[vol_idx+1] - _notch_rot_cen_list[vol_idx]).norm();//TODO: rot center list index

      transX += dist_rot_cen * sin(rot_angY * kDEG2RAD) + dist_rot_cen  * (cos(rot_angZ * kDEG2RAD) - 1.);
      transY += -dist_rot_cen * sin(rot_angX * kDEG2RAD) + dist_rot_cen * sin(rot_angZ * kDEG2RAD);
      transZ += dist_rot_cen  * (cos(rot_angY * kDEG2RAD) - 1.) - dist_rot_cen  * (cos(rot_angX * kDEG2RAD) - 1.);

      auto cur_vol = _snake_vol_list[vol_idx];

      auto cur_rot_cen = _notch_rot_cen_list[vol_idx];
      SetEulerTransform(eul_vol_xform, rot_angX, rot_angY, rot_angZ, transX, transY, transZ, cur_rot_cen);

      // FrameTransform cur_vol_xform = ComputeAffineTransform(eul_vol_xform, transX, transY, transZ);
      // vout << "                  AffineMatrix:\n" << cur_vol_xform.matrix() << std::endl;
      TransformVolume(resampleFilter, cur_vol, eul_vol_xform);
      auto vol_resample = resampleFilter->GetOutput();

      if(vol_idx > 0)
      {
        AddVolumes(addFilter, trans_vol, vol_resample);
        auto trans_vol = addFilter->GetOutput();
      }
      else if(vol_idx == 0)
        trans_vol = vol_resample;
    }

    return trans_vol;
  }
}
