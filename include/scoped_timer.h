#pragma once

#include <string>
#include <chrono>

class ScopedTimer_ {
public:
    explicit ScopedTimer_(const std::string& name);
    ~ScopedTimer_();
    
    static void write_to_csv(const std::string& filename);

private:
    int m_name_id;
    std::chrono::high_resolution_clock::time_point m_start;
};

class NoOpTimer {
public:
    NoOpTimer(const std::string& /*name*/) {}
    static void write_to_csv(const std::string& /*filename*/) {}
};

// =================================================================
// ¡ESTO ES LO QUE FALTA! 
// Define ScopedTimer para que todo el proyecto lo reconozca.
// =================================================================

// Opción 1: Activar los temporizadores (Para medir tiempos y generar el CSV)
using ScopedTimer = ScopedTimer_;

// Opción 2: Desactivar los temporizadores (Si en el futuro quieres que no consuman CPU)
// using ScopedTimer = NoOpTimer;