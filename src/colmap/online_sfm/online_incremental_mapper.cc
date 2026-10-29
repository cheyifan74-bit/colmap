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
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/two_view_geometry.h"
#include "colmap/sensor/rig.h"
#include "colmap/util/logging.h"

#include <algorithm>
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

const Reconstruction& OnlineIncrementalMapper::GetReconstruction() const {
  return mapper_->GetReconstruction();
}

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
  try {
    DatabaseSession session(database_path_);
    return IngestImage(image_id, session.get());
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: ingest image " << image_id
               << " failed: " << e.what();
    return false;
  }
}

bool OnlineIncrementalMapper::IngestImage(const image_t image_id,
                                          Database* database) {
  if (database_cache_->ExistsImage(image_id)) {
    return true;
  }
  if (database == nullptr) {
    return false;
  }
  try {
    if (!database->ExistsImage(image_id) ||
        !database->ExistsKeypoints(image_id)) {
      LOG(WARNING) << "OnlineIncrementalMapper: skip ingest, image "
                   << image_id << " has no keypoints";
      return false;
    }
    return IngestImage(database->ReadImage(image_id), database);
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: ingest image " << image_id
               << " failed: " << e.what();
    return false;
  }
}

bool OnlineIncrementalMapper::IngestImage(class Image image,
                                          Database* database) {
  const image_t image_id = image.ImageId();
  if (database_cache_->ExistsImage(image_id)) {
    return true;
  }
  if (database == nullptr) {
    return false;
  }
  try {
    if (!image.HasCameraId() || !database_cache_->ExistsCamera(image.CameraId())) {
      LOG(ERROR) << "OnlineIncrementalMapper: image " << image_id
                 << " camera not in cache";
      return false;
    }

    const FeatureKeypoints keypoints = database->ReadKeypoints(image_id);
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

  VLOG(2) << "OnlineIncrementalMapper: ingested image " << image_id
          << " into persistent cache (num_images=" << database_cache_->NumImages()
          << ")";
  return true;
}

bool OnlineIncrementalMapper::IngestTwoViewGeometries(const image_t image_id) {
  if (!database_cache_->ExistsImage(image_id) || !database_cache_->CorrespondenceGraph()) {
    return true;
  }
  if (!matcher_) {
    return true;
  }

  // Only check partners from the latest MatchAndWrite / MatchSpecificPairs.
  // Scanning database_cache_->Images() would be O(N) ExistsTwoViewGeometry
  // queries per frame while useful pairs stay ~O(match candidates).
  const std::vector<image_t>& partner_ids = matcher_->LastCandidateIds();
  if (partner_ids.empty()) {
    return true;
  }

  try {
    DatabaseSession session(database_path_);
    auto graph = database_cache_->CorrespondenceGraph();
    const int min_inliers = matching_options_.geometry.min_num_inliers;
    size_t num_ingested = 0;
    for (const image_t prev_id : partner_ids) {
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
      // The pair may arrive after either image was already added to the
      // observation manager (relocalization adds the current frame before its
      // matches exist, and frames that failed to relocalize stay in it). Their
      // cached num_observations would stay at 0 and the next observation on a
      // track that references them aborts the visibility check in
      // IncrementCorrespondenceHasPoint3D. AddImagePair re-reads both counts
      // from the graph, so this is a no-op for the images not registered yet.
      mapper_->NotifyNewImagePair(prev_id, image_id);
      ++num_ingested;
    }
    LOG(INFO) << "OnlineIncrementalMapper: ingest TVG image " << image_id
              << " partners=" << partner_ids.size()
              << " added=" << num_ingested;
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: ingest TVG for image " << image_id
               << " failed: " << e.what();
    return false;
  }
  return true;
}

bool OnlineIncrementalMapper::ResumeFromDisk() {
  if (database_path_.empty() || sparse_path_.empty()) {
    LOG(ERROR) << "OnlineIncrementalMapper: resume needs database and sparse "
                  "paths";
    return false;
  }
  if (database_cache_->NumImages() > 0) {
    LOG(WARNING) << "OnlineIncrementalMapper: resume skipped, cache already "
                    "holds "
                 << database_cache_->NumImages() << " image(s)";
    return true;
  }
  if (loop_closer_ == nullptr || !loop_closer_->IsLoopClosureActive()) {
    LOG(ERROR) << "OnlineIncrementalMapper: resuming a map needs the loop "
                  "closer (MixVPR) for relocalization; set loop_enabled=true "
                  "and a valid loop_mixvpr_engine_path";
    return false;
  }

  std::vector<std::string> registered_names;
  if (!mapper_->LoadSparseMap(&registered_names)) {
    LOG(ERROR) << "OnlineIncrementalMapper: no registered images to resume";
    return false;
  }

  if (!InitDatabaseCache()) {
    return false;
  }

  // Resolve the registered names to database image ids. Frames written to the
  // database but not flushed to sparse (the tail of the previous session) show
  // up here as the difference to db_num_images.
  //
  // One DatabaseSession for the whole restore: opening a session costs ~0.8 ms,
  // which dominated the load time when it was done once per image.
  std::unordered_set<std::string> registered_name_set(registered_names.begin(),
                                                      registered_names.end());
  std::vector<class Image> registered_images;
  size_t db_num_images = 0;
  size_t num_tvgs = 0;
  std::unordered_set<image_t> registered_id_set;
  const size_t min_num_matches =
      static_cast<size_t>(std::max(matching_options_.geometry.min_num_inliers, 0));
  try {
    DatabaseSession session(database_path_);
    Database* database = session.get();

    const std::vector<class Image> images = database->ReadAllImages();
    db_num_images = images.size();
    for (const class Image& image : images) {
      if (registered_name_set.count(image.Name()) > 0) {
        registered_images.push_back(image);
      }
    }
    if (registered_images.size() != registered_names.size()) {
      LOG(ERROR) << "OnlineIncrementalMapper: only "
                 << registered_images.size() << " of "
                 << registered_names.size()
                 << " registered images found in the database, refusing to resume";
      return false;
    }
    std::sort(registered_images.begin(),
              registered_images.end(),
              [](const class Image& a, const class Image& b) {
                return a.ImageId() < b.ImageId();
              });

    // Build the cache incrementally (IngestImage), NOT through
    // DatabaseCache::Load: Load calls CorrespondenceGraph::Finalize(), and a
    // finalized graph cannot accept the images/pairs added while mapping
    // continues.
    for (const class Image& image : registered_images) {
      registered_id_set.insert(image.ImageId());
      if (!IngestImage(image, database)) {
        LOG(ERROR) << "OnlineIncrementalMapper: failed to ingest registered "
                      "image "
                   << image.ImageId();
        return false;
      }
    }

    auto graph = database_cache_->CorrespondenceGraph();
    for (auto& [pair_id, geometry] : database->ReadTwoViewGeometries()) {
      const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
      if (registered_id_set.count(image_id1) == 0 ||
          registered_id_set.count(image_id2) == 0) {
        continue;
      }
      if (geometry.inlier_matches.size() < min_num_matches ||
          geometry.config == TwoViewGeometry::ConfigurationType::DEGENERATE) {
        continue;
      }
      ingested_pairs_.insert(pair_id);
      graph->AddTwoViewGeometry(image_id1, image_id2, std::move(geometry));
      ++num_tvgs;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "OnlineIncrementalMapper: failed to restore the map from the "
                  "database: "
               << e.what();
    return false;
  }

  if (!mapper_->RebuildAfterRestore()) {
    LOG(ERROR) << "OnlineIncrementalMapper: failed to rebuild the mapper for "
                  "the resumed map";
    return false;
  }

  // Local-map projection matching reads the descriptors of the recent
  // registered frames. A normal session caches them frame by frame, so after a
  // resume the previous session's window would be missing and
  // ExtendTracksByProjection would silently find no candidates for the first
  // frames of the new session.
  const size_t desc_window = mapper_->LocalMapWindowSize();
  size_t num_cached_desc = 0;
  if (desc_window > 0 && !registered_images.empty()) {
    const size_t begin = registered_images.size() > desc_window
                             ? registered_images.size() - desc_window
                             : 0;
    try {
      DatabaseSession session(database_path_);
      for (size_t i = begin; i < registered_images.size(); ++i) {
        const image_t image_id = registered_images[i].ImageId();
        if (!session->ExistsDescriptors(image_id)) {
          continue;
        }
        mapper_->CacheDescriptors(
            image_id,
            std::make_shared<const FeatureDescriptors>(
                session->ReadDescriptors(image_id)));
        ++num_cached_desc;
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "OnlineIncrementalMapper: failed to preload local-map "
                      "descriptors: "
                   << e.what();
    }
  }

  mode_ = SessionMode::kLocalizing;
  reloc_attempts_ = 0;
  LOG(INFO) << "OnlineIncrementalMapper: resuming map, waiting for "
               "relocalization (registered="
            << registered_id_set.size() << " db_images=" << db_num_images
            << " unflushed="
            << (db_num_images > registered_id_set.size()
                    ? db_num_images - registered_id_set.size()
                    : 0)
            << " tvgs=" << num_tvgs
            << " cache_images=" << database_cache_->NumImages()
            << " cached_descriptors=" << num_cached_desc << ")";
  return true;
}

void OnlineIncrementalMapper::FlushSparse() {
  if (mapper_ == nullptr) {
    return;
  }
  mapper_->WriteSparse();
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
    mapper_->CacheDescriptors(image_id, extracted.descriptors);
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

  if (mode_ == SessionMode::kLocalizing) {
    return LocalizeCurrent(
        image_id,
        image_abs_path,
        cam_from_world_prior,
        extracted.bitmap.IsEmpty() ? nullptr : &extracted.bitmap);
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
      const OnlineLoopCorrection& correction = loop_closer_->LastCorrection();
      mapper_->AdjustLoopBundle(correction.variable_ids, correction.anchor_ids);
      loop_closer_->FuseAfterLoopBA(mapper_->GetReconstruction(),
                                    mapper_->GetObservationManager());
      mapper_->WriteSparse();
    }
  }

  return true;
}

bool OnlineIncrementalMapper::LocalizeCurrent(
    const image_t image_id,
    const std::string& image_abs_path,
    const std::optional<Rigid3d>& cam_from_world_prior,
    const Bitmap* bitmap) {
  if (loop_closer_ == nullptr || !loop_closer_->IsLoopClosureActive()) {
    LOG(ERROR) << "OnlineIncrementalMapper: resuming a map needs an active "
                  "loop closer (MixVPR) to relocalize";
    return false;
  }
  // Bring the new (pose-less) image into the reconstruction so PnP can read its
  // camera and keypoints.
  if (!mapper_->SyncReconstruction()) {
    return false;
  }

  OnlineLoopPair pair;
  const bool relocalized = loop_closer_->Relocalize(
      image_id,
      *database_cache_,
      mapper_->GetReconstruction(),
      matcher_.get(),
      image_abs_path,
      bitmap,
      &pair);
  if (!relocalized) {
    ++reloc_attempts_;
    LOG(INFO) << "OnlineIncrementalMapper: image " << image_id
              << " not relocalized yet (attempts=" << reloc_attempts_
              << "), map stays unchanged";
    return true;
  }
  if (!cam_from_world_prior.has_value()) {
    LOG(WARNING) << "OnlineIncrementalMapper: image " << image_id
                 << " relocalized but has no VIO prior, cannot chain the map "
                    "frame; skipping";
    return true;
  }

  // The matches found during relocalization carry the map correspondences we
  // need for triangulation and BA.
  if (!IngestTwoViewGeometries(image_id)) {
    return false;
  }
  if (!mapper_->RegisterRelocalizedFrame(image_id,
                                         pair.loop_cam_from_world,
                                         *cam_from_world_prior,
                                         pair.loop_observations)) {
    LOG(WARNING) << "OnlineIncrementalMapper: registering relocalized image "
                 << image_id << " failed";
    return false;
  }

  mode_ = SessionMode::kMapping;
  mapper_->WriteSparse();
  LOG(INFO) << "OnlineIncrementalMapper: relocalized image " << image_id
            << " against map image " << pair.image_id2
            << ", resuming incremental mapping";
  return true;
}

}  // namespace colmap
