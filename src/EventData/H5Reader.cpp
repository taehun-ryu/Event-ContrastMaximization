#include "EventData/H5Reader.h"
#include <H5Cpp.h>
#include <vector>
#include <iostream>

H5Reader::H5Reader(const std::string &fileName)
    : fileName_(fileName), file_(nullptr) {
  try {
    file_ = new H5::H5File(fileName_, H5F_ACC_RDONLY);
  } catch (H5::FileIException &error) {
    error.printErrorStack();
    throw;
  }
}

H5Reader::~H5Reader() {
  if (file_) {
    delete file_;
    file_ = nullptr;
  }
}

EventArray::Ptr H5Reader::readEvents() {
  auto event_array = std::make_shared<EventArray>();

  try {
    // Open the "/events" group
    H5::Group grp = file_->openGroup("/events");

    // Open each 1-D dataset
    H5::DataSet ds_x = grp.openDataSet("xs");
    H5::DataSet ds_y = grp.openDataSet("ys");
    H5::DataSet ds_t = grp.openDataSet("ts");
    H5::DataSet ds_p = grp.openDataSet("ps");

    // Determine number of events (all four should have the same length)
    H5::DataSpace space = ds_x.getSpace();
    hsize_t n;
    space.getSimpleExtentDims(&n, nullptr);
    size_t num_events = static_cast<size_t>(n);

    // Allocate buffers
    std::vector<uint16_t> xs(num_events);
    std::vector<uint16_t> ys(num_events);
    std::vector<double>   ts(num_events);
    std::vector<uint8_t>  ps(num_events);

    // Read into buffers
    ds_x.read(xs.data(), H5::PredType::NATIVE_UINT16);
    ds_y.read(ys.data(), H5::PredType::NATIVE_UINT16);
    ds_t.read(ts.data(), H5::PredType::NATIVE_DOUBLE);
    ds_p.read(ps.data(), H5::PredType::NATIVE_UINT8);

    // Reconstruct EventArray
    for (size_t i = 0; i < num_events; ++i) {
      double t = ts[i];
      int x    = static_cast<int>(xs[i]);
      int y    = static_cast<int>(ys[i]);
      // convert stored 0/1 polarity back to -1/+1
      int p    = (ps[i] ? +1 : -1);

      auto evt = std::make_shared<Event>(t, Event::LocType(x, y), p);
      event_array->addEvent(evt);
    }

    std::cout << "[H5Reader] Loaded " << event_array->size() << " events.\n";

  } catch (H5::Exception &err) {
    err.printErrorStack();
  }

  return event_array;
}
