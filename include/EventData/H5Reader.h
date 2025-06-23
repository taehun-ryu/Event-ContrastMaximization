#pragma once

#include <H5Cpp.h>
#include <iostream>
#include <string>

#include "EventArray.h"

class H5Reader {
public:
  explicit H5Reader(const std::string &fileName);
  ~H5Reader();

  EventArray::Ptr readEvents();

private:
  std::string fileName_;
  H5::H5File *file_;
};
