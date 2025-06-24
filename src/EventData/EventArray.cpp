#include "EventData/EventArray.h"

#include <algorithm>
#include <iostream>

/// @brief Default constructor for Event.
Event::Event() : timestamp_(0.0), location_(LocType::Zero()), polarity_(0) {}

/// @brief Construct an Event with timestamp, position and polarity.
Event::Event(double timestamp, const LocType& location, int polarity)
    : timestamp_(timestamp), location_(location), polarity_(polarity) {}

/// @brief Get timestamp of the event.
/// @return Timestamp in seconds.
double Event::getTimestamp() const {
  return timestamp_;
}

/// @brief Get pixel position of the event.
/// @return Eigen 2D vector of (x, y).
Event::LocType Event::getPosition() const {
  return location_;
}

/// @brief Get polarity of the event.
/// @return +1 or -1 (binary event polarity).
int Event::getPolarity() const {
  return polarity_;
}

/// @brief Set timestamp of the event.
void Event::setTimestamp(double timestamp) {
  timestamp_ = timestamp;
}

/// @brief Set pixel position of the event.
void Event::setLocation(const LocType& location) {
  location_ = location;
}

/// @brief Set polarity of the event.
void Event::setPolarity(int polarity) {
  polarity_ = polarity;
}

/// @brief Default constructor for EventArray.
EventArray::EventArray() {}

/// @brief Construct EventArray from an existing event list.
EventArray::EventArray(const std::vector<Event::Ptr>& events) : events_(events) {}

/// @brief Add an Event to the array.
/// @param event Shared pointer to Event.
void EventArray::addEvent(const Event::Ptr& event) {
  events_.emplace_back(event);
}

/// @brief Add a new Event directly from raw values.
/// @param t Timestamp
/// @param x X position
/// @param y Y position
/// @param p Polarity
void EventArray::addEvent(double t, int x, int y, int p) {
  events_.emplace_back(std::make_shared<Event>(t, Event::LocType(x, y), p));
}

/// @brief Get const reference to all events in the array.
/// @return Reference to vector of Event pointers.
const std::vector<Event::Ptr>& EventArray::getEvents() const {
  return events_;
}

Event::Ptr EventArray::getEvent(size_t idx) const {
  return events_.at(idx);
}

/// @brief Get number of events in the array.
/// @return Event count.
size_t EventArray::size() const {
  return events_.size();
}

/// @brief Normalize timestamps so that first event starts at 0.
void EventArray::normalizeTimestampsToStart() {
  if (events_.empty()) return;

  double t_min = getMinTimestamp();
  for (auto& e : events_) {
    double normalized = e->getTimestamp() - t_min;
    e->setTimestamp(normalized);
  }
}

/// @brief Get the minimum timestamp among all events.
/// @return Earliest timestamp found.
double EventArray::getMinTimestamp() const {
  if (events_.empty()) {
    std::cerr << "[EventArray] getMinTimestamp() called on empty event list.\n";
    return 0.0;
  }

  double min_ts = std::numeric_limits<double>::max();
  for (const auto& e : events_) {
    if (e->getTimestamp() < min_ts) {
      min_ts = e->getTimestamp();
    }
  }
  return min_ts;
}

struct TimeRangeChecker {
  double t_min_, t_max_;
  TimeRangeChecker(double t_min, double t_max) : t_min_(t_min), t_max_(t_max) {}

  bool operator()(const Event::Ptr& e) const { return e->getTimestamp() < t_min_ || e->getTimestamp() > t_max_; }
};

struct CompareByTimestamp {
  bool operator()(const Event::Ptr& a, const Event::Ptr& b) const { return a->getTimestamp() < b->getTimestamp(); }
};

/// @brief Remove all events outside the time interval [t_min, t_max].
/// @param t_min Start time (inclusive)
/// @param t_max End time (inclusive)
void EventArray::filterByTime(double t_min, double t_max) {
  events_.erase(std::remove_if(events_.begin(), events_.end(), TimeRangeChecker(t_min, t_max)), events_.end());
}

/// @brief Sort events in ascending order of timestamp.
void EventArray::sortByTime() {
  std::sort(events_.begin(), events_.end(), CompareByTimestamp());
}

/// @brief Split the event array into fixed time bins.
/// @param bin_size Duration of each bin (in seconds)
/// @return Vector of EventArray pointers, each containing one bin
std::vector<EventArray::Ptr> EventArray::splitByTimeBin(double bin_size) const {
  std::vector<EventArray::Ptr> bins;

  if (events_.empty()) return bins;

  std::vector<Event::Ptr> sorted_events = events_;
  std::sort(sorted_events.begin(), sorted_events.end(), CompareByTimestamp());

  double t_start = sorted_events.front()->getTimestamp();
  double t_end = sorted_events.back()->getTimestamp();
  double t_curr = t_start;

  size_t idx = 0;
  while (t_curr < t_end) {
    double t_next = t_curr + bin_size;

    auto bin = std::make_shared<EventArray>();

    while (idx < sorted_events.size()) {
      double t = sorted_events[idx]->getTimestamp();
      if (t >= t_curr && t < t_next) {
        bin->addEvent(sorted_events[idx]);
        ++idx;
      } else {
        break;
      }
    }

    if (bin->size() > 0) {
      bins.push_back(bin);
    }

    t_curr = t_next;
  }

  return bins;
}
