#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cpphashmap.h"

/**
 * One Billion Row Challenge.
 *
 * - Uses `mmap` to map the file into memory for efficient access.
 * - Since the measurement values are fixed-point doubles with exactly one
 *   decimal, we can use `int64_t` to store the values (multiplied by 10) and
 *   perform the division later when outputting the results.
 * - Handle written parser for measurement values since they lie in
 *   [-99.9, 99.9] range with exactly one decimal place.
 * - `memchr` is used to find line endings efficiently.
 * - The file is processed in parallel using multiple threads, each handling a
 *   chunk of the file. Each thread maintains its own local statistics map which
 *   is later merged into a single map.
 * - Uses a custom FxHasher for better performance with string keys.
 * - Uses open addressing hash map implementation from `cpphashmap.h`.
 */

// Stolen from rustc's FxHasher
// `https://github.com/rust-lang/rustc-hash/blob/5e09ea0a1c7ab7e4f9e27771f5a0e5a36c58d1bb/src/lib.rs`
struct FxHasher {
  static constexpr std::size_t k = 0x517cc1b727220a95;

  std::size_t operator()(std::string_view sv) const noexcept {
    std::size_t hash = 0;

    while (sv.size() >= sizeof(std::size_t)) {
      add_to_hash(hash, *reinterpret_cast<const std::size_t *>(sv.data()));
      sv.remove_prefix(sizeof(std::size_t));
    }

    if (sv.size() >= 4) {
      add_to_hash(hash, *reinterpret_cast<const std::uint32_t *>(sv.data()));
      sv.remove_prefix(4);
    }

    if (sv.size() >= 2) {
      add_to_hash(hash, *reinterpret_cast<const std::uint16_t *>(sv.data()));
      sv.remove_prefix(2);
    }

    if (sv.size() >= 1) {
      add_to_hash(hash, static_cast<std::uint8_t>(sv[0]));
    }

    return hash;
  }

  void add_to_hash(std::size_t &hash, std::size_t i) const {
    // Rotate left by 5
    hash = (hash << 5) | (hash >> (sizeof(std::size_t) * 8 - 5));
    // Xor with i
    hash ^= i;
    // Wrapping multiply
    hash *= k;
  }
};

std::int32_t parse_fixed_point(const char *s) {
  std::int32_t integer_part{};
  bool negative = false;

  // The measurement is in the range [-99.9, 99.9] with exactly one decimal
  // place
  if (*s == '-') {
    negative = true;
    ++s;
  }

  integer_part = static_cast<std::int32_t>(*s++ - '0');
  if (*s != '.') {
    integer_part = integer_part * 10 + static_cast<std::int32_t>(*s - '0');
    ++s;
  }

  // We must now be at the decimal point
  integer_part = integer_part * 10 + static_cast<std::int32_t>(*(s + 1) - '0');
  if (negative) {
    integer_part = -integer_part;
  }

  return integer_part;
}

struct Measurement {
  std::string_view station;
  std::int32_t value;

  static Measurement parse(std::string_view line) {
    constexpr auto delimiter = ';';
    // Minimum length of value is 4: "A;0.0"
    auto delim_pos = line.length() - 4;

    while (line[delim_pos] != delimiter) {
      --delim_pos;
    }

    return Measurement{line.substr(0, delim_pos),
                       parse_fixed_point(line.data() + delim_pos + 1)};
  }
};

struct StationStats {
  std::int32_t min{std::numeric_limits<std::int32_t>::max()};
  std::int32_t max{std::numeric_limits<std::int32_t>::min()};
  std::int64_t sum{0};
  std::uint32_t count{0};

  StationStats &operator+=(const Measurement &m) {
    min = std::min(min, m.value);
    max = std::max(max, m.value);
    sum += m.value;
    ++count;

    return *this;
  }

  StationStats &operator+=(const StationStats &other) {
    min = std::min(min, other.min);
    max = std::max(max, other.max);
    sum += other.sum;
    count += other.count;

    return *this;
  }
};

std::ostream &operator<<(std::ostream &os, const StationStats &stats) {
  os << (stats.min / 10.0f) << '/' << (stats.max / 10.0f) << '/'
     << (stats.sum / (10.0f * stats.count));
  return os;
}

template <typename K, typename V> using FxHashMap = ethical::hash_map<K, V>;

int process(int fd) {
  constexpr auto max_threads = 32;
  auto page_size = sysconf(_SC_PAGESIZE);
  auto nthreads =
      std::min<unsigned int>(std::thread::hardware_concurrency(), max_threads);

  struct stat file_stat {};
  if (fstat(fd, &file_stat) == -1) {
    std::cerr << "Error: Unable to get file size." << std::endl;
    return 1;
  }

  std::size_t file_size = file_stat.st_size;
  if (file_size == 0) {
    std::cerr << "Error: File is empty." << std::endl;
    return 1;
  }

  auto file = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (file == MAP_FAILED) {
    std::cerr << "Error: Unable to map file to memory." << std::endl;
    return 1;
  }
  close(fd);

  const char *file_end = static_cast<const char *>(file) + file_size;
  // std::mutex stdout_mutex{};

  auto worker = [file_end](
                    int i, const char *start, const char *end,
                    FxHashMap<std::string_view, StationStats> &local_stats) {
    // If this is not the first chunk and we are in the middle of a line,
    // find the next newline
    if (i != 0 && start[-1] != '\n') {
      start = static_cast<const char *>(std::memchr(start, '\n', end - start));
      if (start == nullptr) {
        return;
      }
      ++start;
    }

    const char *newline;

    while (start < end && (newline = static_cast<const char *>(std::memchr(
                               start, '\n', file_end - start))) != nullptr) {
      std::string_view line{start, static_cast<std::size_t>(newline - start)};
      start = newline + 1;

      // {
      //   std::lock_guard<std::mutex> lock(stdout_mutex);
      //   std::cout << "Thread " << i << " Processing line: " << line <<
      //   std::endl;
      // }
      Measurement m = Measurement::parse(line);
      local_stats[m.station] += m;
    }
  };

  std::vector<std::thread> threads{};
  std::vector<FxHashMap<std::string_view, StationStats>> results{};
  auto chunk_size = file_size / nthreads;
  auto remainder = file_size % nthreads;

  threads.resize(nthreads);
  results.resize(nthreads);

  for (auto i = 0u; i < nthreads; ++i) {
    auto start = i * chunk_size;
    auto size = chunk_size + (i == nthreads - 1 ? remainder : 0);

    threads[i] = std::thread{worker, i, static_cast<const char *>(file) + start,
                             static_cast<const char *>(file) + start + size,
                             std::ref(results[i])};
  }

  // Collect results
  FxHashMap<std::string_view, StationStats> stats{};
  for (auto i = 0u; i < nthreads; ++i) {
    threads[i].join();

    for (const auto &[station, station_stats] : results[i]) {
      stats[station] += station_stats;
    }
  }

  // Sort the results by station
  std::vector<std::pair<std::string_view, StationStats>> sorted_stats;

  sorted_stats.reserve(stats.size());
  for (const auto &stat : stats) {
    sorted_stats.emplace_back(stat.first, stat.second);
  }
  std::sort(sorted_stats.begin(), sorted_stats.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  auto end = sorted_stats.cend();
  std::cout << "{\n";
  for (auto it = sorted_stats.cbegin(); it != end; ++it) {
    std::cout << "  " << it->first << "=" << it->second;
    if (std::next(it) != end) {
      std::cout << ',';
    }
    std::cout << '\n';
  }
  std::cout << "}\n";

  if (munmap(file, file_size) == -1) {
    std::cerr << "Error: Unable to unmap file from memory." << std::endl;
    return 1;
  }

  return 0;
}

int main(int argc, char *argv[]) {
  std::cin.tie(nullptr);
  std::ios::sync_with_stdio(false);

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  const char *filename = argv[1];
  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    std::cerr << "Error: Unable to open file: " << filename << std::endl;
    return 1;
  }

  return process(fd);
}
