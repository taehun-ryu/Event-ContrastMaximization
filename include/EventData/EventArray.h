#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

class Event {
public:
  using Ptr = std::shared_ptr<Event>;
  using LocType = Eigen::Vector2<std::uint16_t>;

private:
  double timestamp_;
  LocType location_;
  int polarity_;

public:
  Event();
  Event(double timestamp, const LocType &location, int polarity);

  // Getters
  [[nodiscard]] double getTimestamp() const;
  [[nodiscard]] LocType getPosition() const;
  [[nodiscard]] int getPolarity() const;

  // Setters
  void setTimestamp(double timestamp);
  void setLocation(const LocType &location);
  void setPolarity(int polarity);
};

class EventArray {
public:
  using Ptr = std::shared_ptr<EventArray>;

private:
  std::vector<Event::Ptr> events_;

public:
  EventArray();
  explicit EventArray(const std::vector<Event::Ptr> &events);

  void addEvent(const Event::Ptr &event);
  void addEvent(double t, int x, int y, int p);

  [[nodiscard]] const std::vector<Event::Ptr> &getEvents() const;
  [[nodiscard]] Event::Ptr getEvent(size_t idx) const;
  size_t size() const;

  // Timestamp utility
  void normalizeTimestampsToStart();
  double getMinTimestamp() const;
  void filterByTime(double t_min, double t_max);
  void sortByTime();
  std::vector<EventArray::Ptr> splitByTimeBin(double bin_size) const;
};
