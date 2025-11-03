#pragma once

#include <chrono>
#include <filesystem>
#include <string_view>

#define REGISTER_TIMES_
#ifdef REGISTER_TIMES_
constexpr bool REGISTER_TIMES = true;
#else
constexpr bool REGISTER_TIMES = false;
#endif

class ScopedTimer_
{
  public:
    explicit ScopedTimer_(std::string_view name);
    ~ScopedTimer_();

    static void write_to_csv(const std::filesystem::path& filename);

  private:
    int m_name_id;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

class NoOpTimer
{
  public:
    NoOpTimer(std::string_view)
    {
    }
    static void write_to_csv(const std::filesystem::path&)
    {
    }
};

using ScopedTimer = std::conditional_t<REGISTER_TIMES, ScopedTimer_, NoOpTimer>;
