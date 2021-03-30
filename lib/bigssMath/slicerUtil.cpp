
#include "slicerUtil.h"

fiducials read_slicer_fiducials(const std::string &filename, bool ras_to_lps)
{

  std::cout << "Reading fiducials..." << std::endl;
  FILE *file = fopen(filename.c_str(), "r");
  if (!file)
  {
    std::cout << "can't read file " << filename << "!" << std::endl;
    return fiducials();
  }

  int idx = 0;
  double x, y, z = 0.0;
  char label[500];
  char buf[500];

  fiducials fids;

  // read through the first three lines
  for (int i = 0; i < 3; i++)
  {
    fscanf(file, "%[^\n]\n", label);
    std::cout << label << std::endl;
  }

  int i = 0;
  int num_args = fscanf(file, "vtkMRMLMarkupsFiducialNode_%d,%lf,%lf,%lf,%*g,%*g,%*g,%*g,%*d,%*d,%*d,%[^,],,%[^\n]\n",
      &idx, &x, &y, &z, label, buf);

  std::cout << "read " << num_args << std::endl;
 
  while (num_args == 5 || num_args == 6)
  {
    fids.points.conservativeResize(3, i+1);

    fids.points(0, i) = ras_to_lps ? -x : x;
    fids.points(1, i) = ras_to_lps ? -y : y;
    fids.points(2, i) = z;

    std::cout << "fiducial " << i << ": (" << fids.points(0, i) << ", " << fids.points(1, i) << ", " 
      << fids.points(2, i) << "); " << label << std::endl;

    i = i + 1;

    fids.labels.push_back(std::string(label));

    num_args = fscanf(file, "vtkMRMLMarkupsFiducialNode_%d,%lf,%lf,%lf,%*g,%*g,%*g,%*g,%*d,%*d,%*d,%[^,],,%[^\n]\n",
        &idx, &x, &y, &z, label, buf);
  }

  return fids;
}


void write_slicer_fiducials(fiducials fids, std::string filename, bool ras_to_lps)
{
  FILE *file = fopen(filename.c_str(), "w");

  fprintf(file, "# Markups fiducial file version = 4.9\n");
  fprintf(file, "# CoordinateSystem = 0\n");
  fprintf(file, "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n");

  double x, y, z;

  for (int i = 0; i < fids.points.cols(); i++)
  {
    x = ras_to_lps ? -fids.points(0, i) : fids.points(0, i);
    y = ras_to_lps ? -fids.points(1, i) : fids.points(1, i);
    z = fids.points(2, i);

    fprintf(file, "vtkMRMLMarkupsFiducialNode_%d,%lf,%lf,%lf,0,0,0,1,1,1,1,%s,,vtkMRMLScalarVolumeNode1\n",
        i, x, y, z, fids.labels[i].c_str());
  }

  fclose(file);
}


fiducials filter_slicer_fiducials(fiducials fids, std::vector<std::string> labels)
{
  fiducials filtered;
  int num_filtered = 0;
  int num_labels = labels.size();

  for (int i = 0; i < fids.points.cols(); i++)
  {
    for (int j = 0; j < num_labels; j++)
    {
      if (fids.labels[i].find(labels[i]) != std::string::npos)
      {
        filtered.labels.push_back(labels[num_filtered]);
        filtered.points.conservativeResize(3, num_filtered+1);
        num_filtered++;
      }
    }
  }

  return filtered;
}

std::pair<fiducials, fiducials> split_slicer_fiducials(const fiducials &fids, const std::string &s)
{
  int num_fids = fids.labels.size();

  fiducials f1, f2;

  for (int i = 0; i < num_fids; i++)
  {
    if (f1.labels[i].find(s) >= 0)
      f1.add_fiducial(fids.points.col(i), fids.labels[i]);
    else
      f2.add_fiducial(fids.points.col(i), fids.labels[i]);
  }

  return std::pair<fiducials, fiducials>(f1, f2);
}

std::vector<std::string> str_split(const std::string &list, const char &c)
{
  std::vector<std::string> vals;
  std::istringstream f(list);
  std::string s;

  while (getline(f, s, c))
  {
    if (s.size() > 0) vals.push_back(s);
  }

  return vals;
}

