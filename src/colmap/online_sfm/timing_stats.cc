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
#include <iomanip>

namespace colmap {
namespace {

constexpr size_t kLogEveryN = 50;

}  // namespace

TimingStats::TimingStats(std::string output_path,
                         std::string name,
                         std::vector<std::string> extra_columns)
    : output_path_(std::move(output_path)),
      name_(std::move(name)),
      extra_columns_(std::move(extra_columns)) {}

TimingStats::~TimingStats() {
  if (file_.is_open()) {
    file_.flush();
    file_.close();
  }
  WriteSummary();
}

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
  Record(id, ElapsedMilliseconds(), std::move(extra));
}

bool TimingStats::EnsureFileReady() {
  if (output_path_.empty()) {
    return false;
  }
  if (file_.is_open()) {
    return true;
  }

  const std::filesystem::path path(output_path_);
  if (!path.parent_path().empty()) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
  }

  const bool exists =
      std::filesystem::exists(path) && std::filesystem::file_size(path) > 0;
  file_.open(output_path_,
             exists ? (std::ios::out | std::ios::app)
                    : (std::ios::out | std::ios::trunc));
  if (!file_.is_open()) {
    LOG(ERROR) << name_ << ": failed to open " << output_path_;
    return false;
  }

  if (!exists) {
    file_ << "# " << name_ << " timing\n";
    file_ << "# Timed with colmap::Timer (append-only)\n";
    for (const auto& [key, value] : meta_) {
      file_ << "# " << key << "=" << value << "\n";
    }
    file_ << "#\n";
    file_ << "id\telapsed_ms";
    for (const auto& column : extra_columns_) {
      file_ << "\t" << column;
    }
    file_ << "\n";
    header_written_ = true;
  } else {
    header_written_ = true;
  }
  return true;
}

TimingStats::Summary TimingStats::GetSummary() const {
  Summary summary = summary_;
  if (summary.count > 0) {
    summary.avg_ms = summary.sum_ms / static_cast<double>(summary.count);
  } else {
    summary.min_ms = 0.0;
  }
  return summary;
}

void TimingStats::Record(const std::string& id,
                         const double elapsed_ms,
                         std::vector<std::string> extra) {
  extra.resize(extra_columns_.size());

  ++summary_.count;
  summary_.sum_ms += elapsed_ms;
  summary_.min_ms = std::min(summary_.min_ms, elapsed_ms);
  summary_.max_ms = std::max(summary_.max_ms, elapsed_ms);

  if (EnsureFileReady()) {
    file_ << id << "\t" << std::fixed << std::setprecision(3) << elapsed_ms;
    for (const auto& field : extra) {
      file_ << "\t" << field;
    }
    file_ << "\n";
    // Match path records ~15x/frame; avoid forcing a disk sync every line.
    if (summary_.count % kLogEveryN == 0) {
      file_.flush();
    }
  }

  if (summary_.count == 1 || summary_.count % kLogEveryN == 0) {
    const Summary summary = GetSummary();
    LOG(INFO) << std::fixed << std::setprecision(3) << name_
              << ": elapsed_ms=" << elapsed_ms << " avg=" << summary.avg_ms
              << " min=" << summary.min_ms << " max=" << summary.max_ms
              << " n=" << summary.count;
  }
}

void TimingStats::WriteSummary() const {
  if (summary_.count == 0 || output_path_.empty()) {
    return;
  }

  const Summary summary = GetSummary();
  const std::string summary_path = output_path_ + ".summary";
  std::ofstream file(summary_path, std::ios::trunc);
  if (!file.is_open()) {
    LOG(ERROR) << name_ << ": failed to write " << summary_path;
    return;
  }

  file << "# " << name_ << " timing summary\n";
  file << "# Timed with colmap::Timer\n";
  for (const auto& [key, value] : meta_) {
    file << "# " << key << "=" << value << "\n";
  }
  file << std::fixed << std::setprecision(3);
  file << "# count=" << summary.count << "\n";
  file << "# avg_ms=" << summary.avg_ms << "\n";
  file << "# min_ms=" << summary.min_ms << "\n";
  file << "# max_ms=" << summary.max_ms << "\n";
  file << "# sum_ms=" << summary.sum_ms << "\n";
  file << "# tsv=" << output_path_ << "\n";
}

}  // namespace colmap
