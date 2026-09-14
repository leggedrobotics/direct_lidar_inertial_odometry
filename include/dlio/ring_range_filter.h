#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace dlio {

inline constexpr std::size_t kRingRangeCount = 32;

// 10% NIST reflectivity range from the LiDAR channel capability chart.
// Recorded rings are sorted from lowest to highest elevation: ring 0 maps to
// table channel 32 (-55 degrees) and ring 31 maps to channel 1 (+15 degrees).
inline constexpr std::array<double, kRingRangeCount> kRingRangesMeters{{
    10.0,
    20.0, 20.0, 20.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    90.0, 90.0, 90.0,
    90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0,
    90.0, 90.0, 90.0, 90.0, 90.0, 90.0,
}};

inline std::vector<double> defaultRingRangesMeters() {
  return {kRingRangesMeters.begin(), kRingRangesMeters.end()};
}

inline std::array<float, kRingRangeCount> squaredRingRanges(
    const std::vector<double>& ranges_m) {
  if (ranges_m.size() != kRingRangeCount) {
    throw std::invalid_argument("ring range filter requires exactly 32 ranges");
  }

  std::array<float, kRingRangeCount> squared{};
  for (std::size_t ring = 0; ring < ranges_m.size(); ++ring) {
    const double range = ranges_m[ring];
    if (!std::isfinite(range) || range <= 0.0) {
      throw std::invalid_argument("ring ranges must be finite and positive");
    }
    squared[ring] = static_cast<float>(range * range);
  }
  return squared;
}

inline bool pointPassesRingRange(const std::uint16_t ring,
                                 const float x,
                                 const float y,
                                 const float z,
                                 const std::array<float, kRingRangeCount>& squared_ranges) noexcept {
  if (ring >= squared_ranges.size()) {
    return false;
  }
  const float squared_distance = x * x + y * y + z * z;
  return std::isfinite(squared_distance) && squared_distance <= squared_ranges[ring];
}

struct RingRangeFilterResult {
  bool supported_layout = true;
  std::size_t input_points = 0;
  std::size_t output_points = 0;

  std::size_t removedPoints() const noexcept {
    return input_points - output_points;
  }
};

inline RingRangeFilterResult filterPointCloudByRingRange(
    sensor_msgs::msg::PointCloud2& cloud,
    const std::array<float, kRingRangeCount>& squared_ranges) {
  if (cloud.height == 0 || cloud.width == 0) {
    return {};
  }

  const sensor_msgs::msg::PointField* x_field = nullptr;
  const sensor_msgs::msg::PointField* y_field = nullptr;
  const sensor_msgs::msg::PointField* z_field = nullptr;
  const sensor_msgs::msg::PointField* ring_field = nullptr;
  for (const auto& field : cloud.fields) {
    if (field.name == "x") {
      x_field = &field;
    } else if (field.name == "y") {
      y_field = &field;
    } else if (field.name == "z") {
      z_field = &field;
    } else if (field.name == "ring") {
      ring_field = &field;
    }
  }

  const auto is_float32 = [](const sensor_msgs::msg::PointField* field) {
    return field && field->datatype == sensor_msgs::msg::PointField::FLOAT32 &&
           field->count == 1;
  };
  const bool valid_layout =
      is_float32(x_field) && is_float32(y_field) && is_float32(z_field) &&
      ring_field && ring_field->datatype == sensor_msgs::msg::PointField::UINT16 &&
      ring_field->count == 1 && !cloud.is_bigendian && cloud.point_step > 0 &&
      x_field->offset + sizeof(float) <= cloud.point_step &&
      y_field->offset + sizeof(float) <= cloud.point_step &&
      z_field->offset + sizeof(float) <= cloud.point_step &&
      ring_field->offset + sizeof(std::uint16_t) <= cloud.point_step;

  const std::size_t packed_row_size =
      static_cast<std::size_t>(cloud.width) * cloud.point_step;
  const std::size_t required_size =
      (static_cast<std::size_t>(cloud.height) - 1U) * cloud.row_step + packed_row_size;
  if (!valid_layout || cloud.row_step < packed_row_size ||
      cloud.data.size() < required_size) {
    return {false, 0, 0};
  }

  const std::size_t input_count =
      static_cast<std::size_t>(cloud.height) * cloud.width;
  std::size_t output_count = 0;
  for (std::size_t row = 0; row < cloud.height; ++row) {
    const std::size_t row_offset = row * cloud.row_step;
    for (std::size_t column = 0; column < cloud.width; ++column) {
      const std::size_t source_offset = row_offset + column * cloud.point_step;
      const std::uint8_t* source = cloud.data.data() + source_offset;

      float x = 0.0F;
      float y = 0.0F;
      float z = 0.0F;
      std::uint16_t ring = 0;
      std::memcpy(&x, source + x_field->offset, sizeof(x));
      std::memcpy(&y, source + y_field->offset, sizeof(y));
      std::memcpy(&z, source + z_field->offset, sizeof(z));
      std::memcpy(&ring, source + ring_field->offset, sizeof(ring));

      if (!pointPassesRingRange(ring, x, y, z, squared_ranges)) {
        continue;
      }

      const std::size_t destination_offset = output_count * cloud.point_step;
      if (destination_offset != source_offset) {
        std::memmove(cloud.data.data() + destination_offset, source, cloud.point_step);
      }
      ++output_count;
    }
  }

  // Preserve the original organization for the common no-op case. This also
  // guarantees that enabling the filter cannot change downstream behavior
  // when every point is already within its ring limit.
  if (output_count == input_count && cloud.row_step == packed_row_size) {
    return {true, input_count, output_count};
  }

  cloud.height = 1;
  cloud.width = static_cast<std::uint32_t>(output_count);
  cloud.row_step = static_cast<std::uint32_t>(output_count * cloud.point_step);
  cloud.data.resize(cloud.row_step);
  return {true, input_count, output_count};
}

}  // namespace dlio
