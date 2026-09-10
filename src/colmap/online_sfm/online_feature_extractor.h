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

#include "colmap/feature/extractor.h"
#include "colmap/feature/types.h"
#include "colmap/scene/database_cache.h"
#include "colmap/util/types.h"

#include <memory>
#include <string>

namespace colmap {

class TimingStats;

// Reads the image, extracts features if missing, writes keypoints/descriptors,
// and returns in-memory features so matching can skip a database reload.
class OnlineFeatureExtractor {
 public:
  struct ExtractedFeatures {
    std::shared_ptr<const FeatureKeypoints> keypoints;
    std::shared_ptr<const FeatureDescriptors> descriptors;
  };

  OnlineFeatureExtractor(std::string database_path,
                         FeatureExtractionOptions options);
  ~OnlineFeatureExtractor();

  OnlineFeatureExtractor(const OnlineFeatureExtractor&) = delete;
  OnlineFeatureExtractor& operator=(const OnlineFeatureExtractor&) = delete;

  // Returns true if features already exist (skip) or were newly written.
  // On a fresh extract, `features` is filled when non-null.
  bool ExtractAndWrite(image_t image_id,
                       const std::string& image_abs_path,
                       const DatabaseCache& cache,
                       ExtractedFeatures* features = nullptr);

 private:
  bool InitExtractor();

  std::string database_path_;
  FeatureExtractionOptions options_;
  std::unique_ptr<FeatureExtractor> extractor_;
  std::unique_ptr<TimingStats> timing_;
};

}  // namespace colmap
