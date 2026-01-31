/**
 * @file a.cpp
 * @brief NUMA测试方案A - 基本条件变量方案(EXPERT_CG)
 * @details 使用双缓冲+条件变量,每个Buffer有独立的mutex和cv
 * @version 1.00.00
 * @date 2026-01-25
 * @author 技术觉醒团队
 */

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdlib>
#include <climits>

using namespace std;
using namespace std::chrono;

struct Buffer {
    vector<int> data;
    atomic<int> write_pos{0};
    enum class State { WRITABLE, FULL, SENDING };
    State state = State::WRITABLE;
    mutex mtx;
    condition_variable cv;

    Buffer(size_t n) : data(n, 0) {}
};

static int random_delay(int low, int high) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(low, high);
    return dist(rng);
}

void async_send(Buffer* buf, int delay2_ms) {
    {
        unique_lock<mutex> lk(buf->mtx);
        buf->state = Buffer::State::SENDING;
    }
    int d = random_delay(delay2_ms, delay2_ms * 2);
    this_thread::sleep_for(milliseconds(d));

    {
        unique_lock<mutex> lk(buf->mtx);
        std::fill(buf->data.begin(), buf->data.end(), 0);
        buf->write_pos = 0;
        buf->state = Buffer::State::WRITABLE;
    }
    buf->cv.notify_all();
}

void worker_func(int id, Buffer* a, Buffer* b,
                 int delay1_ms, int delay2_ms,
                 int iteration, size_t buf_size, atomic<int>& done_count)
{
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(0, INT_MAX);

    for (int i = 0; i < iteration; ++i) {
        int d1 = random_delay(delay1_ms, delay1_ms * 2);
        this_thread::sleep_for(milliseconds(d1));

        bool written = false;

        while (!written) {
            Buffer* target = nullptr;

            {
                unique_lock<mutex> la(a->mtx, defer_lock);
                unique_lock<mutex> lb(b->mtx, defer_lock);
                lock(la, lb);

                if (a->state == Buffer::State::WRITABLE) target = a;
                else if (b->state == Buffer::State::WRITABLE) target = b;
            }

            if (target) {
                unique_lock<mutex> lk(target->mtx);
                if (target->state != Buffer::State::WRITABLE) continue;

                int pos = target->write_pos.fetch_add(1, memory_order_relaxed);
                if (pos < (int)buf_size) {
                    target->data[pos] = dis(gen);
                    written = true;

                    if (pos + 1 >= (int)buf_size) {
                        target->state = Buffer::State::FULL;
                        lk.unlock();
                        thread(async_send, target, delay2_ms).detach();
                        target->cv.notify_all();
                    }
                }
            } else {
                {
                    unique_lock<mutex> la(a->mtx);
                    auto ok = a->cv.wait_for(la, seconds(5), [&]{ return a->state == Buffer::State::WRITABLE; });
                    if (!ok) {
                        cerr << "[ERROR] Thread " << id << " timeout waiting for buffer A" << endl;
                        exit(1);
                    }
                }
            }
        }
    }
    done_count.fetch_add(1, memory_order_relaxed);
}

int main(int argc, char* argv[]) {
    int delay1 = 3;
    int delay2 = 1000;
    int worker = 12;
    int iteration = 2000;
    size_t buf_size = 512;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--delay1") == 0 && i + 1 < argc)
            delay1 = atoi(argv[++i]);
        else if (strcmp(argv[i], "--delay2") == 0 && i + 1 < argc)
            delay2 = atoi(argv[++i]);
        else if (strcmp(argv[i], "--worker") == 0 && i + 1 < argc)
            worker = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iteration") == 0 && i + 1 < argc)
            iteration = atoi(argv[++i]);
    }

    cout << "=== NUMA Test: Scheme A (Basic Condition Variable) ===" << endl;
    cout << "Configuration:" << endl;
    cout << "  delay1: " << delay1 << " ms" << endl;
    cout << "  delay2: " << delay2 << " ms" << endl;
    cout << "  worker: " << worker << endl;
    cout << "  iteration: " << iteration << endl;
    cout << "  Expected total writes: " << worker * iteration << endl;
    cout << endl;

    Buffer bufA(buf_size), bufB(buf_size);
    atomic<int> done_count{0};

    auto start_time = steady_clock::now();

    vector<thread> threads;
    threads.reserve(worker);
    for (int i = 0; i < worker; ++i)
        threads.emplace_back(worker_func, i, &bufA, &bufB,
                             delay1, delay2, iteration, buf_size, ref(done_count));

    while (done_count.load() < worker) {
        this_thread::sleep_for(milliseconds(200));
    }

    {
        unique_lock<mutex> la(bufA.mtx);
        bufA.cv.wait(la, [&]{ return bufA.state == Buffer::State::WRITABLE; });
    }
    {
        unique_lock<mutex> lb(bufB.mtx);
        bufB.cv.wait(lb, [&]{ return bufB.state == Buffer::State::WRITABLE; });
    }

    auto end_time = steady_clock::now();
    for (auto& t : threads) t.detach();

    auto dur = duration_cast<milliseconds>(end_time - start_time).count();

    cout << "Success! All iterations completed." << endl;
    cout << "Total writes: " << worker * iteration << endl;
    cout << "Total time: " << dur << " ms (" << dur / 1000.0 << " seconds)" << endl;

    return 0;
}
