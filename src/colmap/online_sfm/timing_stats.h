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

#pragma once

#include "colmap/util/timer.h"

#include <string>
#include <utility>
#include <vector>

namespace colmap {

// Reusable stage timer (SIFT / match / BA, ...). One instance per stage.
//
//   TimingStats extract(path, "extract", {"num_features", "image"});
//   extract.Start();
//   ... FeatureExtractor::Extract() ...
//   extract.Record(std::to_string(image_id), {std::to_string(n), image_path});
class TimingStats {
 public:
  TimingStats(std::string output_path,
              std::string name,
              std::vector<std::string> extra_columns = {});
  ~TimingStats();

  TimingStats(const TimingStats&) = delete;
  TimingStats& operator=(const TimingStats&) = delete;

  static std::string FileBesideDatabase(const std::string& database_path,
                                        const std::string& filename);

  void AddMeta(const std::string& key, const std::string& value);

  void Start();
  double ElapsedMilliseconds() const;
  void Record(const std::string& id, std::vector<std::string> extra = {});
  void Record(const std::string& id,
              double elapsed_ms,
              std::vector<std::string> extra = {});

 private:
  struct Sample {
    std::string id;
    double elapsed_ms = 0.0;
    std::vector<std::string> extra;
  };

  struct Summary {
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
  };

  Summary ComputeSummary() const;
  void Write() const;

  std::string output_path_;
  std::string name_;
  std::vector<std::string> extra_columns_;
  std::vector<std::pair<std::string, std::string>> meta_;
  Timer timer_;
  std::vector<Sample> samples_;
};

}  // namespace colmap
