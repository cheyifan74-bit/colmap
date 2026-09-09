// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/online_sfm/timing_stats.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace colmap {

TimingStats::TimingStats(std::string output_path,
                         std::string name,
                         std::vector<std::string> extra_columns)
    : output_path_(std::move(output_path)),
      name_(std::move(name)),
      extra_columns_(std::move(extra_columns)) {}

TimingStats::~TimingStats() { Write(); }

std::string TimingStats::FileBesideDatabase(const std::string& database_path,
                                            const std::string& filename) {
  return (std::filesystem::path(database_path).parent_path() / filename)
      .string();
}

void TimingStats::AddMeta(const std::string& key, const std::string& value) {
  meta_.emplace_back(key, value);
}

void TimingStats::Start() { timer_.Restart(); }

double TimingStats::ElapsedMilliseconds() const {
  return timer_.ElapsedMicroSeconds() / 1000.0;
}

void TimingStats::Record(const std::string& id, std::vector<std::string> extra) {
  extra.resize(extra_columns_.size());
  samples_.push_back({id, ElapsedMilliseconds(), std::move(extra)});

  const Summary summary = ComputeSummary();
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << name_
      << ": elapsed_ms=" << samples_.back().elapsed_ms
      << " avg=" << summary.avg_ms << " min=" << summary.min_ms
      << " max=" << summary.max_ms << " n=" << samples_.size();
  LOG(INFO) << oss.str();
  Write();
}

TimingStats::Summary TimingStats::ComputeSummary() const {
  Summary summary;
  if (samples_.empty()) {
    return summary;
  }
  summary.min_ms = std::numeric_limits<double>::infinity();
  summary.max_ms = 0.0;
  double sum_ms = 0.0;
  for (const auto& sample : samples_) {
    summary.min_ms = std::min(summary.min_ms, sample.elapsed_ms);
    summary.max_ms = std::max(summary.max_ms, sample.elapsed_ms);
    sum_ms += sample.elapsed_ms;
  }
  summary.avg_ms = sum_ms / static_cast<double>(samples_.size());
  return summary;
}

void TimingStats::Write() const {
  if (samples_.empty() || output_path_.empty()) {
    return;
  }

  const Summary summary = ComputeSummary();
  std::ofstream file(output_path_, std::ios::trunc);
  if (!file.is_open()) {
    LOG(ERROR) << name_ << ": failed to write " << output_path_;
    return;
  }

  file << "# " << name_ << " timing\n";
  file << "# Timed with colmap::Timer\n";
  for (const auto& [key, value] : meta_) {
    file << "# " << key << "=" << value << "\n";
  }
  file << "# count=" << samples_.size() << "\n";
  file << std::fixed << std::setprecision(3);
  file << "# avg_ms=" << summary.avg_ms << "\n";
  file << "# min_ms=" << summary.min_ms << "\n";
  file << "# max_ms=" << summary.max_ms << "\n";
  file << "#\n";
  file << "id\telapsed_ms";
  for (const auto& column : extra_columns_) {
    file << "\t" << column;
  }
  file << "\n";
  for (const auto& sample : samples_) {
    file << sample.id << "\t" << sample.elapsed_ms;
    for (const auto& field : sample.extra) {
      file << "\t" << field;
    }
    file << "\n";
  }
}

}  // namespace colmap
