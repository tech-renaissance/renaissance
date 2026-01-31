/**
 * @file b.cpp
 * @brief NUMA测试方案B - 原子操作+条件变量方案(EXPERT_OP)
 * @details 使用原子变量+条件变量,减少锁竞争
 * @version 1.00.00
 * @date 2026-01-25
 * @author 技术觉醒团队
 */

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <climits>
#include <string>

using namespace std;

struct Config {
    int delay1 = 3;
    int delay2 = 1000;
    int worker = 12;
    int iteration = 2000;
};

constexpr int ARRAY_SIZE = 512;
constexpr int TIMEOUT_SECONDS = 5;

struct Buffer {
    vector<int> data;
    atomic<int> write_count{0};
    atomic<bool> is_transferring{false};
    mutex mtx;
    condition_variable cv_transfer_done;
    condition_variable cv_buffer_full;

    Buffer() : data(ARRAY_SIZE, 0) {}

    void reset() {
        fill(data.begin(), data.end(), 0);
        write_count.store(0, memory_order_release);
        is_transferring.store(false, memory_order_release);
    }
};

struct GlobalState {
    Buffer buffers[2];
    atomic<int> current_buffer{0};
    atomic<bool> program_running{true};
    atomic<int> total_writes{0};
    atomic<int> threads_completed{0};

    Config config;

    mutex global_mtx;
    condition_variable cv_all_done;
};

GlobalState g_state;

thread_local mt19937 t_rng(random_device{}());

int getRandomDelay(int base_delay) {
    uniform_real_distribution<double> dist(1.0, 2.0);
    return static_cast<int>(base_delay * dist(t_rng));
}

void sleepMs(int ms) {
    this_thread::sleep_for(chrono::milliseconds(ms));
}

Config parseArgs(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--delay1" && i + 1 < argc) {
            cfg.delay1 = stoi(argv[++i]);
        } else if (arg == "--delay2" && i + 1 < argc) {
            cfg.delay2 = stoi(argv[++i]);
        } else if (arg == "--worker" && i + 1 < argc) {
            cfg.worker = stoi(argv[++i]);
        } else if (arg == "--iteration" && i + 1 < argc) {
            cfg.iteration = stoi(argv[++i]);
        }
    }
    return cfg;
}

void asyncTransfer(int buffer_idx, int delay2) {
    Buffer& buf = g_state.buffers[buffer_idx];

    int transfer_time = getRandomDelay(delay2);
    sleepMs(transfer_time);

    {
        lock_guard<mutex> lock(buf.mtx);
        fill(buf.data.begin(), buf.data.end(), 0);
        buf.write_count.store(0, memory_order_release);
        buf.is_transferring.store(false, memory_order_release);
    }
    buf.cv_transfer_done.notify_all();
}

bool tryTriggerTransfer(int buffer_idx) {
    Buffer& buf = g_state.buffers[buffer_idx];

    if (buf.write_count.load(memory_order_acquire) >= ARRAY_SIZE) {
        bool expected = false;
        if (buf.is_transferring.compare_exchange_strong(expected, true,
                memory_order_acq_rel, memory_order_acquire)) {
            thread(asyncTransfer, buffer_idx, g_state.config.delay2).detach();
            buf.cv_buffer_full.notify_all();
            return true;
        }
    }
    return false;
}

bool waitBufferWritable(int buffer_idx, int timeout_sec) {
    Buffer& buf = g_state.buffers[buffer_idx];

    unique_lock<mutex> lock(buf.mtx);
    auto deadline = chrono::steady_clock::now() + chrono::seconds(timeout_sec);

    while (buf.is_transferring.load(memory_order_acquire) ||
           buf.write_count.load(memory_order_acquire) >= ARRAY_SIZE) {

        if (!g_state.program_running.load(memory_order_acquire)) {
            return false;
        }

        if (buf.cv_transfer_done.wait_until(lock, deadline) == cv_status::timeout) {
            return false;
        }
    }
    return true;
}

void workerThread(int thread_id) {
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(0, INT_MAX);

    int writes_done = 0;
    int target_writes = g_state.config.iteration;
    int delay1 = g_state.config.delay1;

    while (writes_done < target_writes && g_state.program_running.load(memory_order_acquire)) {
        int preprocess_time = getRandomDelay(delay1);
        sleepMs(preprocess_time);

        if (!g_state.program_running.load(memory_order_acquire)) {
            break;
        }

        bool write_success = false;

        while (!write_success && g_state.program_running.load(memory_order_acquire)) {
            int buf_idx = g_state.current_buffer.load(memory_order_acquire);
            Buffer& current_buf = g_state.buffers[buf_idx];

            if (current_buf.is_transferring.load(memory_order_acquire)) {
                int other_idx = 1 - buf_idx;
                Buffer& other_buf = g_state.buffers[other_idx];

                if (!other_buf.is_transferring.load(memory_order_acquire)) {
                    g_state.current_buffer.compare_exchange_weak(buf_idx, other_idx,
                        memory_order_acq_rel, memory_order_acquire);
                    continue;
                }

                if (!waitBufferWritable(buf_idx, TIMEOUT_SECONDS)) {
                    if (g_state.program_running.load(memory_order_acquire)) {
                        cerr << "[ERROR] Timeout waiting for buffer " << buf_idx
                                  << " (thread " << thread_id << ")" << endl;
                        g_state.program_running.store(false, memory_order_release);
                    }
                    return;
                }
                continue;
            }

            int write_pos = current_buf.write_count.fetch_add(1, memory_order_acq_rel);

            if (write_pos >= ARRAY_SIZE) {
                current_buf.write_count.fetch_sub(1, memory_order_acq_rel);
                tryTriggerTransfer(buf_idx);

                int other_idx = 1 - buf_idx;
                g_state.current_buffer.compare_exchange_weak(buf_idx, other_idx,
                    memory_order_acq_rel, memory_order_acquire);
                continue;
            }

            int value = dis(gen);
            current_buf.data[write_pos] = value;
            write_success = true;
            writes_done++;
            g_state.total_writes.fetch_add(1, memory_order_relaxed);

            if (write_pos + 1 >= ARRAY_SIZE) {
                bool expected = false;
                if (current_buf.is_transferring.compare_exchange_strong(expected, true,
                        memory_order_acq_rel, memory_order_acquire)) {
                    thread(asyncTransfer, buf_idx, g_state.config.delay2).detach();
                    int other_idx = 1 - buf_idx;
                    g_state.current_buffer.compare_exchange_weak(buf_idx, other_idx,
                        memory_order_acq_rel, memory_order_acquire);
                }
            }
        }
    }

    int completed = g_state.threads_completed.fetch_add(1, memory_order_acq_rel) + 1;
    if (completed >= g_state.config.worker) {
        g_state.cv_all_done.notify_all();
    }
}

void finalTransfer() {
    for (int i = 0; i < 2; i++) {
        Buffer& buf = g_state.buffers[i];
        if (buf.write_count.load(memory_order_acquire) > 0 &&
            !buf.is_transferring.load(memory_order_acquire)) {
            buf.is_transferring.store(true, memory_order_release);
            int transfer_time = getRandomDelay(g_state.config.delay2);
            sleepMs(transfer_time);
            buf.reset();
        } else if (buf.is_transferring.load(memory_order_acquire)) {
            unique_lock<mutex> lock(buf.mtx);
            buf.cv_transfer_done.wait_for(lock, chrono::seconds(TIMEOUT_SECONDS), [&]() {
                return !buf.is_transferring.load(memory_order_acquire);
            });
        }
    }
}

int main(int argc, char* argv[]) {
    g_state.config = parseArgs(argc, argv);

    cout << "=== NUMA Test: Scheme B (Atomic Operations) ===" << endl;
    cout << "Configuration:" << endl;
    cout << "  delay1: " << g_state.config.delay1 << " ms" << endl;
    cout << "  delay2: " << g_state.config.delay2 << " ms" << endl;
    cout << "  worker: " << g_state.config.worker << endl;
    cout << "  iteration: " << g_state.config.iteration << endl;
    cout << "  Expected total writes: " << g_state.config.worker * g_state.config.iteration << endl;
    cout << endl;

    auto start_time = chrono::steady_clock::now();

    vector<thread> workers;
    workers.reserve(g_state.config.worker);

    for (int i = 0; i < g_state.config.worker; i++) {
        workers.emplace_back(workerThread, i);
    }

    {
        unique_lock<mutex> lock(g_state.global_mtx);

        int expected_writes_per_buffer = (g_state.config.worker * g_state.config.iteration + ARRAY_SIZE - 1) / ARRAY_SIZE;
        int max_time_sec = (g_state.config.delay1 * 2 * g_state.config.iteration +
                          g_state.config.delay2 * 2 * expected_writes_per_buffer) / 1000 + 60;

        bool completed = g_state.cv_all_done.wait_for(lock, chrono::seconds(max_time_sec), []() {
            return g_state.threads_completed.load(memory_order_acquire) >= g_state.config.worker ||
                   !g_state.program_running.load(memory_order_acquire);
        });

        if (!completed && g_state.program_running.load(memory_order_acquire)) {
            cerr << "[ERROR] Overall timeout!" << endl;
            g_state.program_running.store(false, memory_order_release);
        }
    }

    g_state.program_running.store(false, memory_order_release);

    for (int i = 0; i < 2; i++) {
        g_state.buffers[i].cv_transfer_done.notify_all();
        g_state.buffers[i].cv_buffer_full.notify_all();
    }

    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }

    if (g_state.total_writes.load(memory_order_acquire) <
        g_state.config.worker * g_state.config.iteration) {
        cerr << "[ERROR] Not all writes completed. Total writes: "
                  << g_state.total_writes.load(memory_order_acquire)
                  << " / " << g_state.config.worker * g_state.config.iteration << endl;
        return 1;
    }

    finalTransfer();

    auto end_time = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "Success! All iterations completed." << endl;
    cout << "Total writes: " << g_state.total_writes.load(memory_order_acquire) << endl;
    cout << "Total time: " << duration.count() << " ms ("
              << duration.count() / 1000.0 << " seconds)" << endl;

    return 0;
}
