#include <iostream>
#include <ctime>
#include "Timer.h"

static clock_t timer_start;

// Functions for use in other programs
extern "C" {
    void start_timer() {
        timer_start = clock();
    }
    
    double stop_timer() {
        clock_t timer_end = clock();
        return static_cast<double>(timer_end - timer_start) / CLOCKS_PER_SEC;
    }
}

// Separate program for timer testing
#ifdef TEST_TIMER
int main() {
    start_timer();
    // Test workload
    volatile double sum = 0;
    for(int i = 0; i < 1'000'000; i++) {
        sum += i * 0.1;
    }
    double elapsed = stop_timer();
    
    std::cout << "Elapsed time: " << elapsed << "s\n";
    std::cout << "I am FASTEEER!" << std::endl;
    return 0;
}
#endif