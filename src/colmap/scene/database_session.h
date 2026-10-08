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

#pragma once

#include "colmap/scene/database.h"

#include <filesystem>
#include <memory>
#include <mutex>

namespace colmap {

// Short-lived RAII database connection serialized by canonical path.
// Same path (after weakly_canonical) never has two Open()s at once.
// Path mutexes are never erased so returned references stay valid.
class DatabaseSession {
 public:
  explicit DatabaseSession(const std::filesystem::path& path);
  ~DatabaseSession();

  DatabaseSession(const DatabaseSession&) = delete;
  DatabaseSession& operator=(const DatabaseSession&) = delete;
  DatabaseSession(DatabaseSession&&) = delete;
  DatabaseSession& operator=(DatabaseSession&&) = delete;

  Database& operator*() { return *database_; }
  const Database& operator*() const { return *database_; }
  Database* operator->() { return database_.get(); }
  const Database* operator->() const { return database_.get(); }
  Database* get() { return database_.get(); }
  const Database* get() const { return database_.get(); }

 private:
  std::unique_lock<std::mutex> lock_;
  std::shared_ptr<Database> database_;
};

}  // namespace colmap
