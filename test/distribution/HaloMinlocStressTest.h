/*
 * (C) Copyright 2024 UCAR.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 */

#ifndef TEST_DISTRIBUTION_HALO_MINLOC_STRESS_TEST_H_
#define TEST_DISTRIBUTION_HALO_MINLOC_STRESS_TEST_H_

#include <vector>
#include <utility>
#include <limits>
#include <chrono>
#include <algorithm>

#define ECKIT_TESTING_SELF_REGISTER_CASES 0

#include "eckit/mpi/Comm.h"
#include "eckit/testing/Test.h"

#include "oops/mpi/mpi.h"
#include "oops/runs/Test.h"
#include "oops/util/Logger.h"

namespace ioda {
namespace test {

// -----------------------------------------------------------------------------
/*!
 * \brief Halo 分發的 minloc 壓力測試
 * 
 * \details 模擬 Halo.cc 中的實際情況，使用 64 個進程和大量觀測數據
 * 來驗證是否為 MPI 函式庫問題
 */
// -----------------------------------------------------------------------------

// 以分批方式執行 allReduce(minloc) 以避免大型向量在高進程數下觸發 MPI 錯誤
static void batchedAllReduceMinloc(const eckit::mpi::Comm & comm,
                                   const std::vector<std::pair<double, int>> & local,
                                   std::vector<std::pair<double, int>> & global,
                                   const size_t batchSizeHint) {
  const size_t total = local.size();
  if (global.size() != total) global.resize(total);

  // 動態決定批次大小（至少 1000），並限制不超過總量
  size_t batchSize = batchSizeHint;
  if (batchSize == 0) {
    const size_t heuristic = std::max(static_cast<size_t>(1000), total / (static_cast<size_t>(comm.size()) * 4 + 1));
    batchSize = std::min(heuristic, total == 0 ? static_cast<size_t>(1) : total);
  }
  batchSize = std::max(static_cast<size_t>(1), std::min(batchSize, total));

  const size_t numBatches = (total + batchSize - 1) / batchSize;
  oops::Log::debug() << "使用分批 allReduce(minloc): 總數=" << total
                     << ", 批次大小=" << batchSize
                     << ", 批次數=" << numBatches << std::endl;

  for (size_t b = 0; b < numBatches; ++b) {
    const size_t start = b * batchSize;
    const size_t end = std::min(start + batchSize, total);
    const size_t thisBatch = end - start;

    std::vector<std::pair<double, int>> locBatch(thisBatch);
    std::vector<std::pair<double, int>> glbBatch(thisBatch);

    for (size_t i = 0; i < thisBatch; ++i) {
      locBatch[i] = local[start + i];
    }

    comm.allReduce(locBatch, glbBatch, eckit::mpi::minloc());

    for (size_t i = 0; i < thisBatch; ++i) {
      global[start + i] = glbBatch[i];
    }
  }
}

void testHaloMinlocStressTest() {
  const eckit::mpi::Comm & comm = oops::mpi::world();
  const int myRank = comm.rank();
  const int commSize = comm.size();
  
  oops::Log::debug() << "Halo 壓力測試開始 - Rank " << myRank 
                     << " of " << commSize << std::endl;

  // 模擬 Halo.cc 中的實際數據規模
  // 使用較大的向量來模擬實際的觀測數據
  const size_t nglocs = 10000;  // 模擬 10000 個觀測位置
  
  std::vector<std::pair<double, int>> dist_and_lidx_loc(nglocs);
  std::vector<std::pair<double, int>> dist_and_lidx_glb(nglocs);
  
  // 初始化為無窮大，模擬 Halo.cc 中的初始化
  const double inf = std::numeric_limits<double>::infinity();
  for (size_t jj = 0; jj < nglocs; ++jj) {
    dist_and_lidx_loc[jj] = std::make_pair(inf, myRank);
  }
  
  // 為每個進程設定一些實際的距離值
  // 模擬 Halo 分發中的觀測數據分配
  for (size_t jj = 0; jj < nglocs; ++jj) {
    if (jj % commSize == static_cast<size_t>(myRank)) {
      // 這個進程負責這個觀測位置
      double distance = 1.0 + (jj % 100) * 0.1;  // 1.0 到 10.9
      dist_and_lidx_loc[jj] = std::make_pair(distance, myRank);
    }
  }
  
  // 模擬 Halo.cc 中的調試輸出
  oops::Log::debug() << "Rank " << myRank << " - 數據大小=" << dist_and_lidx_loc.size() 
                     << " elements, 總大小=" << (dist_and_lidx_loc.size() * sizeof(std::pair<double, int>)) 
                     << " bytes" << std::endl;
  
  // 記錄開始時間
  auto start_time = std::chrono::high_resolution_clock::now();

  // 以分批方式執行 allReduce(minloc) 以驗證可修復性
  const size_t batchSizeHint = 2000;  // 可依需要調整或做成參數
  oops::Log::debug() << "Rank " << myRank << " - 以分批方式開始 allReduce minloc 操作" << std::endl;
  batchedAllReduceMinloc(comm, dist_and_lidx_loc, dist_and_lidx_glb, batchSizeHint);
  oops::Log::debug() << "Rank " << myRank << " - 分批 allReduce minloc 操作完成" << std::endl;

  // 記錄結束時間
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  
  oops::Log::debug() << "Rank " << myRank << " - allReduce 耗時: " << duration.count() << " ms" << std::endl;
  
  // 驗證結果
  size_t valid_results = 0;
  for (size_t jj = 0; jj < nglocs; ++jj) {
    if (dist_and_lidx_glb[jj].first < inf) {
      valid_results++;
    }
  }
  
  oops::Log::debug() << "Rank " << myRank << " - 有效結果數量: " << valid_results << std::endl;
  
  // 基本驗證：確保操作成功完成
  EXPECT(valid_results > 0);
  
  oops::Log::debug() << "Rank " << myRank << " - Halo 壓力測試通過" << std::endl;
}

// -----------------------------------------------------------------------------

class HaloMinlocStressTest : public oops::Test {
 public:
  HaloMinlocStressTest() {}
  virtual ~HaloMinlocStressTest() {}
  
 private:
  std::string testid() const override {return "test::HaloMinlocStressTest";}

  void register_tests() const override {
    std::vector<eckit::testing::Test>& ts = eckit::testing::specification();

    ts.emplace_back(CASE("distribution/HaloMinlocStressTest/testHaloMinlocStressTest")
      { testHaloMinlocStressTest(); });
  }

  void clear() const override {}
};

// -----------------------------------------------------------------------------

}  // namespace test
}  // namespace ioda

#endif  // TEST_DISTRIBUTION_HALO_MINLOC_STRESS_TEST_H_



