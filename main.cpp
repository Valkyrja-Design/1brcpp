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
 * - memchr for finding newlines
 * - FxHasher for hashing strings
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

struct Measurement {
  std::string_view station;
  std::int16_t value;

  static Measurement parse(std::string_view line) {
    constexpr auto delimiter = ';';
    auto delim_pos = line.length() - 1;

    // Temperature value is of the form [-]DD.D
    std::int16_t value{static_cast<std::int16_t>(line[delim_pos] - '0')};
    value += static_cast<std::int16_t>(line[delim_pos - 2] - '0') * 10;
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
  os << (stats.min / 10.0f) << '/' << (stats.max / 10.0f) << '/'
     << (stats.sum / (10.0f * stats.count));
  return os;
}

template <typename K, typename V>
using FxHashMap = std::unordered_map<K, V, FxHasher>;

int process(int fd) {
  constexpr auto max_threads = 1;
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
  madvise(file, file_size, MADV_SEQUENTIAL);
  close(fd);

  std::string_view file_view{static_cast<const char *>(file), file_size};
  const char *newline;
  // TODO: Allocate keys on heap (and inline small strings) to prevent `mdavise`
  // from being useless because of `string_view`s
  FxHashMap<std::string_view, StationStats> stats{};
  stats.reserve(10000);

  while ((newline = static_cast<const char *>(std::memchr(
              file_view.data(), '\n', file_view.size()))) != nullptr) {
    auto line = file_view.substr(0, newline - file_view.data());
    if (line.empty()) {
      break;
    }

    file_view.remove_prefix(line.size() + 1);

    Measurement m = Measurement::parse(line);
    stats[m.station] += m;
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
