#pragma once
#include "backend.h"

// Simple Device Singleton
class Device {
public:
    static Backend* get() {
        static CPUBackend cpu_backend; 
        return &cpu_backend;
    }
};
