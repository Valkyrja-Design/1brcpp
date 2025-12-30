#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <immintrin.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/**
 * One Billion Row Challenge.
 *
 * - mmap the file into memory
 * - Since the measurement values are fixed-point doubles with exactly one
 *   decimal, we can use `int64_t` to store the values (multiplied by 10) and
 *   perform the division later when outputting the results
 * - Handle written parser for measurement values since they lie in
 *   [-99.9, 99.9] range with exactly one decimal place
 * - SIMD for finding newlines
 * - FxHasher for hashing strings
 * - Branchless temperature value parsing
 */

// rustc's FxHasher
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

struct Measurement {
  std::string_view station;
  std::int16_t value;

  // Assumes the line is at least 32 bytes long
  static Measurement parse_simd(std::string_view line) {
    // Credit: https://curiouscoding.nl/posts/1brc/
    constexpr auto delimiter = ';';

    // Find the delimiter position
    __m256i bytes =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(line.data()));
    int mask = _mm256_movemask_epi8(
        _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(delimiter)));
    int delim_pos = __builtin_ctz(mask);
    int pos = delim_pos + 1;
    int sign = line[pos] == '-' ? -1 : 1;
    pos += line[pos] == '-';

    // Reads `bc.d` or `c.d?`
    std::uint32_t value{};
    std::memcpy(&value, line.data() + pos, 4);

    // Remove `?`
    value <<= 8 * (4 - (line.size() - pos));

    // Convert ascii to their integer values. ascii digits have the upper 4 bits
    // as 0b0011
    value &= 0x0f000f0f;

    // value is now
    //                       0d    00    0c    0b
    //                        d     0     c     d  | * 1
    //         10d     0    10c   10b              | * 10 << 16
    //   100d    0  100c   100b                    | * 100 << 24
    //   100d  10d  100c   want   10b    c     d   | sum
    // The sum we need is at the 4th byte but since the max value of sum 999 >
    // 256 it won't fit in that byte alone. The cool thing to notice is that the
    // adjacent byte (100c) is always divisible by 4 and so its last 2 bits are
    // always 0. Therefore, we can take those 2 bits and our bytes == 10 bits
    // which can easily fit in all the valid values
    constexpr uint64_t c = 1 + (10 << 16) + (100 << 24);
    // Take the lower 10 digits after multiplication
    value = ((value * c) >> 24) & ((1 << 10) - 1);

    return Measurement{
        line.substr(0, delim_pos),
        static_cast<std::int16_t>(sign * static_cast<int>(value))};
  }

  static Measurement parse(std::string_view line) {
    constexpr auto delimiter = ';';
    auto delim_pos = line.length() - 1;

    // Temperature value is of the form [-]DD.D
    std::int16_t value{static_cast<std::int16_t>(
        (line[delim_pos] - '0') + (line[delim_pos - 2] - '0') * 10)};
    delim_pos -= 3;

    if (line[delim_pos] != delimiter && line[delim_pos] != '-') {
      value += static_cast<std::int16_t>(line[delim_pos] - '0') * 100;
      --delim_pos;
    }

    if (line[delim_pos] == '-') {
      value = -value;
      --delim_pos;
    }

    return Measurement{line.substr(0, delim_pos), value};
  }
};

struct StationStats {
  std::int16_t min{std::numeric_limits<std::int16_t>::max()};
  std::int16_t max{std::numeric_limits<std::int16_t>::min()};
  std::uint32_t count{0};
  std::int64_t sum{0};

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
  os << std::setprecision(1) << std::fixed << (stats.min / 10.0f) << '/'
     << (stats.max / 10.0f) << '/' << (stats.sum / (10.0f * stats.count));
  return os;
}

template <typename K, typename V>
using FxHashMap = std::unordered_map<K, V, FxHasher>;

int process(int fd) {
  auto page_size = sysconf(_SC_PAGESIZE);
  auto nthreads = std::thread::hardware_concurrency();

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
  madvise(file, file_size, MADV_SEQUENTIAL | MADV_WILLNEED);
  close(fd);

  const char *file_end = static_cast<const char *>(file) + file_size;

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

    while (start + 32 <= end) {
      // SIMD on 32 bytes
      __m256i bytes =
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(start));
      int mask = _mm256_movemask_epi8(
          _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8('\n')));

      if (mask == 0) {
        start += 32;
        continue;
      }

      const char *newline = start + __builtin_ctz(mask);
      Measurement m = Measurement::parse_simd(
          {start, static_cast<std::size_t>(newline - start)});
      local_stats[m.station] += m;
      start = newline + 1;
    }

    // Remaining bytes
    while (start < end) {
      const char *newline = start;
      // Manually search for newline (avoids memchr function call overhead). The
      // challenge promised a newline at the end of every line.
      while (*newline != '\n')
        ++newline;

      Measurement m = Measurement::parse(
          {start, static_cast<std::size_t>(newline - start)});
      local_stats[m.station] += m;
      start = newline + 1;
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

    results[i].reserve(1 << 14);
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
