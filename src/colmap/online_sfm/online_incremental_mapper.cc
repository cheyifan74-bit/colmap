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

#include "colmap/online_sfm/online_incremental_mapper.h"

#include "colmap/feature/sift.h"
#include "colmap/scene/correspondence_graph.h"
#include "colmap/scene/database_session.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/logging.h"

#include <exception>
#include <vector>

namespace colmap {

OnlineIncrementalMapper::OnlineIncrementalMapper(
    std::string database_path,
    FeatureExtractionOptions extraction_options,
    OnlineMatchingOptions matching_options,
    std::string sparse_path,
    OnlineMapperOptions mapper_options,
    OnlineLoopCloserOptions loop_options)
    : database_path_(std::move(database_path)),
      sparse_path_(std::move(sparse_path)),
      matching_options_(matching_options) {
  if (extraction_options.sift) {
    matching_options_.max_num_features_hint =
        extraction_options.sift->max_num_features;
  }
  database_cache_ = std::make_shared<DatabaseCache>();
  extractor_ = std::make_unique<OnlineFeatureExtractor>(
      database_path_, std::move(extraction_options));
  matcher_ = std::make_unique<OnlineFeatureMatcher>(
      database_path_, matching_options_);
  mapper_ = std::make_unique<OnlineMapper>(
      database_cache_, sparse_path_, std::move(mapper_options));
  loop_closer_ = std::make_unique<OnlineLoopCloser>(database_path_,
                                                    std::move(loop_options));
}

OnlineIncrementalMapper::~OnlineIncrementalMapper() = default;

const std::vector<OnlineLoopPair>& OnlineIncrementalMapper::ConfirmedLoops()
    const {
  static const std::vector<OnlineLoopPair> kEmpty;
  return loop_closer_ ? loop_closer_->ConfirmedLoops() : kEmpty;
}

bool OnlineIncrementalMapper::InitDatabaseCache() {
  if (database_cache_->NumCameras() > 0) {
    return true;
  }
  return LoadCamerasFromDatabase();
}

bool OnlineIncrementalMapper::LoadCamerasFromDatabase() {
  try {
    DatabaseSession session(database_path_);
    for (auto& camera : session->ReadAllCameras()) {
      if (database_cache_->ExistsCamera(camera.camera_id)) {
        continue;
      }
      if (!database_cache_->ExistsRig(camera.camera_id)) {
        class Rig rig;
        rig.SetRigId(camera.camera_id);
        rig.AddRefSensor(camera.SensorId());
        database_cache_->AddRig(std::move(rig));
      }
      database_cache_->AddCamera(std::move(camera));
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: failed to load cameras: "
               << e.what();
    return false;
  }
  if (database_cache_->NumCameras() == 0) {
    LOG(ERROR) << "OnlineIncrementalMapper: database has no cameras";
    return false;
  }
  LOG(INFO) << "OnlineIncrementalMapper: loaded " << database_cache_->NumCameras()
            << " camera(s) into database cache";
  return true;
}

bool OnlineIncrementalMapper::IngestImage(const image_t image_id) {
  if (database_cache_->ExistsImage(image_id)) {
    return true;
  }

  try {
    DatabaseSession session(database_path_);
    if (!session->ExistsImage(image_id) ||
        !session->ExistsKeypoints(image_id)) {
      LOG(WARNING) << "OnlineIncrementalMapper: skip ingest, image "
                   << image_id << " has no keypoints";
      return false;
    }

    class Image image = session->ReadImage(image_id);
    if (!image.HasCameraId() || !database_cache_->ExistsCamera(image.CameraId())) {
      LOG(ERROR) << "OnlineIncrementalMapper: image " << image_id
                 << " camera not in cache";
      return false;
    }

    const FeatureKeypoints keypoints = session->ReadKeypoints(image_id);
    if (keypoints.empty()) {
      LOG(WARNING) << "OnlineIncrementalMapper: skip ingest, image "
                   << image_id << " has empty keypoints";
      return false;
    }
    std::vector<Eigen::Vector2d> points(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
      points[i] = Eigen::Vector2d(keypoints[i].x, keypoints[i].y);
    }
    image.SetPoints2D(points);

    const camera_t camera_id = image.CameraId();
    if (!image.HasFrameId()) {
      image.SetFrameId(image.ImageId());
    }
    if (!database_cache_->ExistsFrame(image.FrameId())) {
      class Frame frame;
      frame.SetFrameId(image.FrameId());
      frame.SetRigId(camera_id);
      frame.AddDataId(image.DataId());
      database_cache_->AddFrame(std::move(frame));
    }
    database_cache_->AddImage(std::move(image));
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: ingest image " << image_id
               << " failed: " << e.what();
    return false;
  }

  LOG(INFO) << "OnlineIncrementalMapper: ingested image " << image_id
            << " into persistent cache (num_images=" << database_cache_->NumImages()
            << ")";
  return true;
}

bool OnlineIncrementalMapper::IngestTwoViewGeometries(const image_t image_id) {
  if (!database_cache_->ExistsImage(image_id) || !database_cache_->CorrespondenceGraph()) {
    return true;
  }

  try {
    DatabaseSession session(database_path_);
    auto graph = database_cache_->CorrespondenceGraph();
    const int min_inliers = matching_options_.geometry.min_num_inliers;
    for (const auto& [prev_id, image] : database_cache_->Images()) {
      (void)image;
      if (prev_id == image_id) {
        continue;
      }
      const image_pair_t pair_id = ImagePairToPairId(prev_id, image_id);
      if (ingested_pairs_.count(pair_id) > 0) {
        continue;
      }
      if (!session->ExistsTwoViewGeometry(prev_id, image_id)) {
        continue;
      }
      TwoViewGeometry geometry =
          session->ReadTwoViewGeometry(prev_id, image_id);
      ingested_pairs_.insert(pair_id);
      if (geometry.inlier_matches.size() <
          static_cast<size_t>(min_inliers)) {
        continue;
      }
      if (geometry.config == TwoViewGeometry::ConfigurationType::DEGENERATE) {
        continue;
      }
      graph->AddTwoViewGeometry(prev_id, image_id, std::move(geometry));
      // mapper_->NotifyNewImagePair(prev_id, image_id);
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: ingest TVG for image " << image_id
               << " failed: " << e.what();
    return false;
  }
  return true;
}

bool OnlineIncrementalMapper::Process(
    const image_t image_id, const std::string& image_abs_path,
    const std::optional<Rigid3d>& cam_from_world_prior) {
  if (database_path_.empty()) {
    LOG(ERROR) << "OnlineIncrementalMapper: empty database path";
    return false;
  }
  if (image_id == kInvalidImageId) {
    LOG(ERROR) << "OnlineIncrementalMapper: invalid image_id";
    return false;
  }
  if (!InitDatabaseCache()) {
    return false;
  }

  last_image_id_ = image_id;
  last_image_abs_path_ = image_abs_path;
  last_cam_from_world_prior_ = cam_from_world_prior;
  if (cam_from_world_prior.has_value()) {
    matcher_->PutPosePrior(image_id, *cam_from_world_prior);
  }

  OnlineFeatureExtractor::ExtractedFeatures extracted;
  if (!extractor_->ExtractAndWrite(
          image_id, image_abs_path, *database_cache_, &extracted)) {
    return false;
  }
  if (extracted.keypoints && extracted.descriptors) {
    matcher_->CacheFeatures(
        image_id, extracted.keypoints, extracted.descriptors);
  }

  if (!IngestImage(image_id)) {
    LOG(WARNING) << "OnlineIncrementalMapper: image " << image_id
                 << " not ingested, skip match/map";
    return true;
  }

  if (!matcher_->MatchAndWrite(image_id, *database_cache_)) {
    return false;
  }
  if (!IngestTwoViewGeometries(image_id)) {
    return false;
  }

  if (!mapper_->MappingCurrentImage(image_id, cam_from_world_prior)) {
    return false;
  }

  if (loop_closer_ && loop_closer_->IsLoopClosureActive()) {
    const bool corrected = loop_closer_->Process(
        image_id,
        image_abs_path,
        *database_cache_,
        mapper_->GetReconstruction(),
        mapper_->GetObservationManager(),
        matcher_.get(),
        extracted.bitmap.IsEmpty() ? nullptr : &extracted.bitmap);
    if (corrected) {
      mapper_->WriteSparse();
    }
  }

  return true;
}

}  // namespace colmap
