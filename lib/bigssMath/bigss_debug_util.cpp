#include "bigss_debug_util.h"

int bigss::time_now_ms()
{
  std::chrono::time_point<std::chrono::high_resolution_clock> time
    = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    time.time_since_epoch()).count();
}
