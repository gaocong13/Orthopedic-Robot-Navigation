#ifndef SLICER_UTIL_H
#define SLICER_UTIL_H

#include <string>
#include <vector>
#include <iostream>

#include <Eigen/Dense>

enum fids_class {
  FRAGMENT = 0,
  ILIUM,
  INJECTED,
  GLUED,
  LEFT,
  RIGHT
};

struct fiducials
{
  Eigen::Matrix3Xd points;
  std::vector<std::string> labels;

  fiducials() : points(), labels() {}

  fiducials(const Eigen::Matrix3Xd p) : points(p)
  {
    for (int i = 0; i < p.cols(); i++) labels.push_back(std::to_string(i));
  }

  fiducials(int n)
  {
    for (int i = 0; i < n; i++) labels.push_back(std::to_string(i));
    points = Eigen::Matrix3Xd::Zero(3, n);
  }

  void add_fiducial(const Eigen::Vector3d &p, const std::string &l)
  {
    int num_fids = labels.size();
    points.conservativeResize(3, num_fids + 1);
    points.col(num_fids) = p;
    labels.push_back(l);
  }

  void set_label(const int &i, const std::string &l)
  {
    int num_fids = labels.size();
    if (i >= num_fids)
    {
      return;
    }
    labels[i] = l;
  }

   friend std::ostream& operator<< (std::ostream& stream, const fiducials &fids)
   {
      int num_fids = fids.labels.size();
      for (int i = 0; i < num_fids; i++)
      {
        stream << fids.labels[i] << ": (" << fids.points(0, i) << ", " << fids.points(1, i)
          << ", " << fids.points(2, i) << ")" << std::endl;
      }
      return stream;
   }
};

fiducials read_slicer_fiducials(const std::string &filename, bool ras_to_lps=true);
void write_slicer_fiducials(fiducials fids, std::string filename, bool ras_to_lps=true);
fiducials filter_slicer_fiducials(fiducials fids, std::vector<std::string> labels);

/* Splits slicer fiducials struct.
 * If they contain the parameter string, they are returned in the first part of the
 * pair, if not, in the second.
 * \param fids Fiducials to split
 * \param s String that they contain.
 * \return std::pair that contains fiducials containing string in first slot and complement
 * in the second.
 */
std::pair<fiducials, fiducials> split_slicer_fiducials(const fiducials &fids, const std::string &s);

std::vector<std::string> str_split(const std::string &list, const char &c=',');

#endif

