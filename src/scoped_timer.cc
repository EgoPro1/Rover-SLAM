#include "scoped_timer.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Implementation details hidden inside anonymous namespace
namespace
{

struct TimerRecord
{
    int name_id;
    int64_t start_ns;
    int64_t end_ns;
};

// Global storage for all timer records
std::mutex g_records_mutex;
std::vector<TimerRecord> g_records;

// Name → id mapping to save memory usage in records
std::mutex g_name_map_mutex;
std::unordered_map<std::string, int> g_name_to_id;
std::vector<std::string> g_id_to_name;
std::atomic<int> g_next_id{0};

// Cambiado de std::string_view a const std::string&
int get_or_create_name_id(const std::string& name)
{
    std::lock_guard<std::mutex> lock(g_name_map_mutex);
    auto it = g_name_to_id.find(name);
    if (it != g_name_to_id.end())
    {
        return it->second;
    }

    int id = g_next_id++;
    g_name_to_id.emplace(name, id);
    g_id_to_name.push_back(name);
    return id;
}

} // namespace

// Cambiado de std::string_view a const std::string&
ScopedTimer_::ScopedTimer_(const std::string& name)
    : m_name_id(get_or_create_name_id(name)), m_start(std::chrono::high_resolution_clock::now())
{
}

ScopedTimer_::~ScopedTimer_()
{
    auto end = std::chrono::high_resolution_clock::now();
    int64_t start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(m_start.time_since_epoch()).count();
    int64_t end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();

    std::lock_guard<std::mutex> lock(g_records_mutex);
    g_records.push_back({m_name_id, start_ns, end_ns});
}

// Cambiado de const std::filesystem::path& a const std::string&
void ScopedTimer_::write_to_csv(const std::string& filename)
{
    std::cout << "Start writing processing time to csv " << std::endl;
    std::ofstream ofile(filename);
    if (!ofile.is_open())
    {
        std::cerr << "Could not open output file: " << filename << std::endl;
        return;
    }

    ofile << "name,start_ns,end_ns,duration_ms\n";

    std::lock_guard<std::mutex> lock(g_records_mutex);
    if (g_records.empty())
    {
        std::cerr << "No timing records to write" << std::endl;
        return;
    }
    for (const auto& record : g_records)
    {
        const std::string& name = g_id_to_name[record.name_id];
        double duration_ms = (record.end_ns - record.start_ns) * 1e-6;
        ofile << name << "," << record.start_ns << "," << record.end_ns << "," << duration_ms << "\n";
    }
    std::cout << "End of writting processing time to " << filename << std::endl;
}