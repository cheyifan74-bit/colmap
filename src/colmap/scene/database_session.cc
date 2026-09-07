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
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/scene/database_session.h"

#include <unordered_map>

namespace colmap {
namespace {

std::string CanonicalDatabasePath(const std::filesystem::path& path) {
  std::error_code ec;
  const std::filesystem::path canonical =
      std::filesystem::weakly_canonical(path, ec);
  if (ec) {
    return std::filesystem::absolute(path).lexically_normal().string();
  }
  return canonical.string();
}

std::mutex& MutexForDatabasePath(const std::string& canonical_path) {
  static std::mutex registry_mutex;
  static std::unordered_map<std::string, std::unique_ptr<std::mutex>>
      path_mutexes;
  std::lock_guard<std::mutex> registry_lock(registry_mutex);
  std::unique_ptr<std::mutex>& slot = path_mutexes[canonical_path];
  if (!slot) {
    slot = std::make_unique<std::mutex>();
  }
  return *slot;
}

}  // namespace

DatabaseSession::DatabaseSession(const std::filesystem::path& path)
    : lock_(MutexForDatabasePath(CanonicalDatabasePath(path))),
      database_(Database::Open(path)) {}

DatabaseSession::~DatabaseSession() {
  if (database_) {
    database_->Close();
    database_.reset();
  }
}

}  // namespace colmap
