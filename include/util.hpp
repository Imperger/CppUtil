#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <ratio>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif
namespace util
{
/*
 * mean
 */
template<typename It>
inline double mean(It begin, It end)
{
    assert(begin != end && "Mean of empty set is undefined");

    return std::accumulate(begin, end, 0.0) / std::distance(begin, end);
}
/*
 * median
 */
template<typename It>
double median(It begin, It end)
{
    assert(begin != end && "Median of empty set is undefined");

    size_t size = std::distance(begin, end);

    std::nth_element(begin, std::next(begin, size / 2), end);

    return size % 2 == 0 ? (*std::next(begin, size / 2 - 1) + *std::next(begin, size / 2)) / 2. :
                           *std::next(begin, size / 2);
}
/*
 * variance
 */
template<typename It>
inline double variance(It begin, It end)
{
    double mean = util::mean(begin, end);
    return std::accumulate(begin, end, 0.0, [&](double acc, double x) { return acc + std::pow(x - mean, 2); }) /
           static_cast<double>(std::distance(begin, end));
}
/*
 * standard_deviation
 */
template<typename It>
inline double standard_deviation(It begin, It end)
{
    return std::sqrt(variance(begin, end));
}
template<typename It>
struct max_subarray_result
{
    It begin;
    It end;
    typename It::value_type sum;
};
/*
 * max_subarray_sum
 */
template<typename It>
max_subarray_result<It> max_subarray_sum(It begin, It end)
{
    if (begin == end)
        return {begin, end, 0};

    It second = std::next(begin);
    max_subarray_result<It> best{begin, second, *begin};
    for (max_subarray_result<It> cur = best; cur.end != end; ++cur.end)
    {
        if (cur.sum <= 0)
        {
            cur.begin = cur.end;
            cur.sum = *cur.end;
        }
        else
        {
            cur.sum += *cur.end;
        }

        if (cur.sum > best.sum)
        {
            best.begin = cur.begin;
            best.end = std::next(cur.end);
            best.sum = cur.sum;
        }
    }

    return best;
}

/*
 * matrix
 */
template<typename T>
class matrix
{
    class row_holder
    {
      public:
        explicit row_holder(std::size_t m, matrix &mt) : m(m), mt(mt) {}

        T &operator[](std::size_t n) { return mt.data[m * mt.n + n]; }

      private:
        std::size_t m;
        matrix &mt;
    };
    class const_row_holder
    {
      public:
        explicit const_row_holder(std::size_t m, const matrix &mt) : m(m), mt(mt) {}

        const T &operator[](std::size_t n) const { return mt.data[m * mt.m + n]; }

      private:
        std::size_t m;
        const matrix &mt;
    };

  public:
    explicit matrix(std::size_t m, std::size_t n) : m(m), n(n), data(m * n) {}

    template<typename Type, size_t M, size_t N>
    explicit matrix(const Type (&mtx)[M][N]) : m(M), n(N)
    {
        data.reserve(m * n);
        for (std::size_t m = 0; m < M; ++m)
        {
            for (std::size_t n = 0; n < N; ++n)
            {
                data.push_back(mtx[m][n]);
            }
        }
    }
    row_holder operator[](std::size_t idx) { return row_holder(idx, *this); }

    const_row_holder operator[](std::size_t idx) const { return const_row_holder(idx, *this); }

    T determinant() const
    {
        if (m != n)
            throw std::runtime_error("Determinant operation available only on square matrix");

        return calc_determinant(*this);
    }

  private:
    static T calc_determinant(const matrix &m)
    {
        if (m.m == 1)
        {
            return m[0][0];
        }
        else if (m.m == 2)
        {
            return m[0][0] * m[1][1] - m[0][1] * m[1][0];
        }
        else
        {
            T det = 0;
            for (int p = 0; p < m.m; p++)
            {
                matrix temp(m.m - 1, m.n - 1);

                for (std::size_t n = m.n, i = 0; n < m.data.size(); ++n)
                {
                    if (n % m.n == p)
                        continue;

                    temp.data[i++] = m.data[n];
                }

                det += m[0][p] * static_cast<T>(pow(-1, p)) * calc_determinant(temp);
            }
            return det;
        }
    }

  private:
    std::size_t m;
    std::size_t n;
    std::vector<T> data;
};
template<typename Type, size_t M, size_t N>
matrix(const Type (&mtx)[M][N]) -> matrix<Type>;

/*
 * Longest common subsequence
 */
template<typename RndIt1, typename RndIt2>
std::vector<RndIt1> longest_common_subsequence(RndIt1 first1, RndIt1 last1, RndIt2 first2, RndIt2 last2)
{
    enum dir
    {
        left,
        up,
        diagonal
    };

    auto const length1 = last1 - first1;
    auto const length2 = last2 - first2;

    assert(length1 >= 0 && "Negative input1 length");
    assert(length2 >= 0 && "Negative input2 length");

    if (!(length1 && length2))
        return {};

    matrix<int64_t> lens(length1 + 1, length2 + 1);

    for (int64_t m = length1 - 1; m >= 0; --m)
    {
        for (int64_t n = length2 - 1; n >= 0; --n)
        {
            if (*(first1 + m) == *(first2 + n))
            {
                lens[m][n] = lens[m + 1][n + 1] + 1;
            }
            else
            {
                lens[m][n] = std::max(lens[m + 1][n], lens[m][n + 1]);
            }
        }
    }

    std::vector<RndIt1> ret;
    for (size_t m = 0, n = 0; lens[m][n] != 0;)
    {
        if (*(first1 + m) == *(first2 + n))
        {
            ret.push_back(first1 + m);
            ++m;
            ++n;
        }
        else
        {
            if (lens[m][n] == lens[m + 1][n])
            {
                ++m;
            }
            else
            {
                ++n;
            }
        }
    }

    return ret;
}

/*
 * Longest common subsequence
 */
template<typename RndIt1, typename RndIt2>
std::pair<RndIt1, RndIt1> longest_common_substring(RndIt1 first1, RndIt1 last1, RndIt2 first2, RndIt2 last2)
{
    auto const length1 = last1 - first1;
    auto const length2 = last2 - first2;

    assert(length1 >= 0 && "Negative input1 length");
    assert(length2 >= 0 && "Negative input2 length");

    if (!(length1 && length2))
        return {};

    matrix<int64_t> lens(length1 + 1, length2 + 1);
    int64_t max_len = 0;
    int64_t max_start = length1;
    for (int64_t m = length1 - 1; m >= 0; --m)
    {
        for (int64_t n = length2 - 1; n >= 0; --n)
        {
            if (*(first1 + m) == *(first2 + n))
            {
                int64_t len = lens[m + 1][n + 1] + 1;
                lens[m][n] = len;

                if (len > max_len)
                {
                    max_len = len;
                    max_start = m;
                }
            }
        }
    }

    return std::make_pair(first1 + max_start, first1 + max_start + max_len);
}

/*
 * Longest palindrome subsequence
 */
template<typename RndIt>
std::vector<RndIt> longest_palindrome_subsequence(RndIt first, RndIt last)
{
    return longest_common_subsequence(first, last, std::make_reverse_iterator(last), std::make_reverse_iterator(first));
}

/*
 * queue
 */
template<typename T>
class queue
{
  public:
    queue() : mem{nullptr, 0}, head(0), size(0) {}

    ~queue() { delete[] mem.ptr; }

    template<typename V>
    void push(V &&val)
    {
        if (has_slot())
        {
            assign_val(std::forward<V>(val));
        }
        else
        {
            extend();

            assign_val(std::forward<V>(val));
        }

        ++size;
    }

    void pop()
    {
        if (size == 0)
            return;

        head = (head + 1) % mem.size;
        --size;
    }

    T const &front() const
    {
        if (size == 0)
            throw std::runtime_error("Trying to access front element on empty queue");

        return *(mem.ptr + head);
    }

    T &front() { return const_cast<T &>(static_cast<const queue &>(*this).front()); }

    T &back() { return const_cast<T &>(static_cast<const queue &>(*this).back()); }

    T const &back() const

    {
        if (size == 0)
            throw std::runtime_error("Trying to access last element on empty queue");

        return *(mem.ptr + (head + size - 1) % mem.size);
    }

    bool empty() const { return !size; }

  private:
    bool has_slot() const { return size < mem.size; }

    T *free_slot() const { return mem.ptr + (head + size) % mem.size; }

    void extend()
    {
        size_t const new_size = mem.size ? 2 * mem.size : 1;
        T *new_mem = new T[new_size];

        std::copy(mem.ptr, mem.ptr + mem.size, new_mem);

        delete[] mem.ptr;

        mem.ptr = new_mem;
        mem.size = new_size;
    }

    void assign_val(T const &val) { *free_slot() = val; }

    void assign_val(T &&val) { *free_slot() = std::move(val); }

    struct
    {
        T *ptr;
        size_t size;
    } mem;
    size_t head;
    size_t size;
};

/*
 * fixed_queue
 */
template<typename T, size_t N>
class fixed_queue
{
  public:
    bool enqueue(const T &value)
    {
        if (full())
            return false;

        data[tail] = value;
        tail = next_n(tail);

        return true;
    }

    T dequeue()
    {
        if (empty())
            throw std::runtime_error("Queue is empty");

        const T &ret = data[head];
        head = next_n(head);

        return ret;
    }

    inline bool empty() const { return head == tail; }

    inline bool full() const { return next_n(tail) == head; }

    inline size_t capacity() const { return data.size() - 1; }

    inline size_t size() const { return head < tail ? tail - head : data.size() - head + tail; }

  private:
    inline size_t next_n(size_t n) const { return (n + 1) % data.size(); }

  private:
    size_t head = 0;
    size_t tail = 0;
    std::array<T, N + 1> data;
};
/*
 * threadsafe_queue
 */
template<typename T>
class threadsafe_queue
{
  public:
    void push(const T &val)
    {
        std::lock_guard<std::shared_mutex> lk(m);
        queue.push(val);
    }

    T pop()
    {
        std::lock_guard<std::shared_mutex> lk(m);
        T val = queue.front();
        queue.pop();
        return val;
    }

    std::optional<T> try_pop()
    {
        std::lock_guard<std::shared_mutex> lk(m);

        if (queue.empty())
            return {};

        T val = queue.front();
        queue.pop();
        return val;
    }

    bool empty() const
    {
        std::shared_lock<std::shared_mutex> lk(m);
        return queue.empty();
    }

    size_t size() const
    {
        std::shared_lock<std::shared_mutex> lk(m);
        return queue.size();
    }

  private:
    mutable std::shared_mutex m;
    std::queue<T> queue;
};
/*
 * threadsafe_priority_queue
 */
template<typename T>
class threadsafe_priority_queue
{
  public:
    void push(const T &val)
    {
        std::lock_guard<std::shared_mutex> lk(m);
        queue.push(val);
    }

    T pop()
    {
        std::lock_guard<std::shared_mutex> lk(m);
        T val = queue.top();
        queue.pop();
        return val;
    }

    std::optional<T> try_pop()
    {
        std::lock_guard<std::shared_mutex> lk(m);

        if (queue.empty())
            return {};

        T val = queue.top();
        queue.pop();
        return val;
    }

    T &top() { return queue.top(); }
    bool empty() const
    {
        std::shared_lock<std::shared_mutex> lk(m);
        return queue.empty();
    }

    size_t size() const
    {
        std::shared_lock<std::shared_mutex> lk(m);
        return queue.size();
    }

  private:
    mutable std::shared_mutex m;
    std::priority_queue<T> queue;
};
/**
 * task_package
 */
class task_package
{
    friend class thread_pool;

  public:
    template<typename Fn, typename... Args>
    void append(Fn fn, Args... args)
    {
        if (sealed)
            throw std::runtime_error("Append a task to an already scheduled package is forbidden");

        tasks.push_back(std::bind(&task_package::task_wrapper,
                                  this,
                                  static_cast<typename decltype(tasks)::value_type>(std::bind(fn, args...))));
    }

    void wait()
    {
        std::unique_lock<std::mutex> lk(m);
        complete_event.wait(lk, [this] { return completed(); });
    }

    bool completed() const { return completion_counter == tasks.size(); }

  private:
    void task_wrapper(std::function<void()> fn)
    {
        fn();
        {
            std::lock_guard<std::mutex> lk(m);
            ++completion_counter;

            if (completed())
                complete_event.notify_all();
        }
    }

  private:
    std::vector<std::function<void()>> tasks;
    uint64_t completion_counter = 0;
    std::condition_variable complete_event;
    std::mutex m;
    bool sealed = false;
};

/**
 * thread_pool
 */
class thread_pool
{
    struct worker
    {
        uint64_t id = 0;
        std::atomic<bool> is_busy;
        std::function<void()> task;
        std::thread t;
        std::mutex m;
        std::condition_variable waiter;
    };

  public:
    explicit thread_pool(size_t t = std::thread::hardware_concurrency()) : isRunning(true), pool(t)
    {
        for (size_t n = 0; n < pool.size(); ++n)
        {
            pool[n].id = n;
            pool[n].is_busy = false;
            pool[n].t = std::thread(&thread_pool::worker_loop, this, std::ref(pool[n]));
        }
    }

    template<typename Fn, typename... Args>
    void schedule(Fn fn, Args... args)
    {
        auto candidate = find_free_worker();
        if (candidate)
        {
            worker *&c = *candidate;
            std::lock_guard<std::mutex> lk(c->m);
            if (!c->is_busy)
            {
                c->task = std::bind(fn, args...);
                c->is_busy = true;
                c->waiter.notify_one();
            }
        }
        else
        {
            waitList.push(std::bind(fn, args...));
        }
    }

    void schedule(task_package &p)
    {
        p.sealed = true;

        for (auto t : p.tasks)
            schedule(t);
    }

    bool is_busy()
    {
        return !waitList.empty() ||
               std::any_of(pool.begin(), pool.end(), [](const worker &x) { return (bool)x.is_busy; });
    }
    uint64_t executed_tasks() const { return task_counter; }

    uint64_t size() const { return pool.size(); }

    uint64_t queue_length() const { return waitList.size(); }

    void release()
    {
        isRunning = false;
        join();
    }

    void join()
    {
        for (worker &x : pool)
        {
            x.waiter.notify_one();
            x.t.join();
        }
    }

    ~thread_pool() { release(); }

  private:
    void worker_loop(worker &self)
    {
        while (isRunning)
        {
            if (self.task)
            {
                self.is_busy = true;
                self.task();
                ++task_counter;
                self.task = {};
            }

            if (auto task = waitList.try_pop())
            {
                self.task = *task;
                continue;
            }

            self.is_busy = false;

            std::unique_lock<std::mutex> lk(self.m);
            self.waiter.wait(lk, [&] { return self.task || !isRunning; });
        }

#ifdef _DEBUG
        std::cout << "Thread " << self.id << " exited\n";
#endif
    }

    std::optional<worker *> find_free_worker()
    {
        auto it = std::find_if(pool.begin(), pool.end(), [&](const worker &x) { return !x.is_busy; });

        if (it == pool.end())
            return {};

        return &*it;
    }

  private:
    std::atomic<bool> isRunning;
    std::vector<worker> pool;
    threadsafe_queue<std::function<void()>> waitList;
    std::atomic<uint64_t> task_counter = 0;
};
/*
 * parallel_map
 */
template<typename Container>
class parallel_map
{
    using value_type = typename Container::value_type;
    using transform_type = std::function<value_type(const value_type &)>;
    using iterator = typename Container::iterator;

  public:
    explicit parallel_map(Container &target, thread_pool &pool) : target(&target), pool(&pool) {}

    template<typename Fn, typename... Args>
    parallel_map &map(Fn fn, Args... args) &
    {
        transformList.push_back(std::bind(fn, std::placeholders::_1, args...));
        return *this;
    }

    template<typename Fn, typename... Args>
    parallel_map &&map(Fn fn, Args... args) &&
    {
        transformList.push_back(std::bind(fn, std::placeholders::_1, args...));
        return std::forward<parallel_map>(*this);
    }

    void run() &
    {
        auto chunk_size = target->size() / pool->size();
        for (auto begin = target->begin(); begin < target->end(); std::advance(begin, chunk_size))
        {
            auto end = begin + chunk_size;
            if (end > target->end())
                end = target->end();

            pkg.append([this, begin, end]() {
                for (auto x : transformList)
                {
                    std::transform(begin, end, begin, x);
                }
            });
        }

        pool->schedule(pkg);
    }

    void wrun()
    {
        run();
        pkg.wait();
    }

  private:
    Container *target;
    thread_pool *pool;
    task_package pkg;
    std::vector<transform_type> transformList;
};
/**
 * timer class
 */
class timer
{
  public:
    using clock = std::chrono::steady_clock;
    void start() { tm = clock::now(); }

    template<typename Period = std::chrono::milliseconds>
    uint64_t stop()
    {
        return std::chrono::duration_cast<Period>(clock::now() - tm).count();
    }

    template<typename Period = std::chrono::milliseconds>
    uint64_t reset()
    {
        auto x = stop<Period>();
        start();
        return x;
    }

  private:
    std::chrono::steady_clock::time_point tm;
};

template<typename Fn, typename... Args>
struct has_return_type
{
    static constexpr bool value =
        std::negation_v<std::bool_constant<std::is_void_v<std::invoke_result_t<Fn, Args...>>>>;
};

std::string ms_to_string(uint64_t ms, int precision = 2)
{
    double x = static_cast<double>(ms);
    std::ostringstream os;
    os.precision(precision);
    os << std::fixed;

    if (ms < 1000)
    {
        os << x << "ms";
    }
    else if (ms < 60 * 1000)
    {
        os << x / 1000 << "s";
    }
    else if (ms < 60 * 60 * 1000)
    {
        os << x / 60 / 1000 << "m";
    }
    else
    {
        os << x / 60 / 60 / 1000 << "h";
    }

    return os.str();
}

#ifdef TIMING
namespace internal
{
class initialized_timer : protected timer
{
    std::chrono::steady_clock::time_point start_time;

  public:
    using timer::reset;
    using timer::start;
    using timer::stop;
    initialized_timer() : start_time(timer::clock::now()) {}
    template<typename Period = std::chrono::milliseconds>
    uint64_t elapsed() const
    {
        return std::chrono::duration_cast<Period>(clock::now() - start_time).count();
    }
};
inline initialized_timer timer;
} // namespace internal

template<typename Fn, typename... Args, typename = std::enable_if_t<has_return_type<Fn, Args...>::value>>
auto timing(const char *msg, std::ostream &os, Fn &&fn, Args &&... args)
{
    os << "[" << ms_to_string(internal::timer.elapsed(), 1) << "] " << msg;
    internal::timer.start();
    auto &&ret = fn(args...);
    uint64_t d = internal::timer.stop();
    os << " +" << ms_to_string(d) << '\n';
    return ret;
}

template<typename Fn,
         typename... Args,
         typename = std::enable_if_t<std::negation_v<std::bool_constant<has_return_type<Fn, Args...>::value>>>>
void timing(const char *msg, std::ostream &os, Fn &&fn, Args &&... args)
{
    os << "[" << ms_to_string(internal::timer.elapsed(), 1) << "] " << msg;
    internal::timer.start();
    fn(args...);
    uint64_t d = internal::timer.stop();
    os << " +" << ms_to_string(d) << '\n';
}

#endif // TIMING

struct PrintOptions
{
    size_t width;
    bool title;
    bool ascii;
    char no_ascii_placeholder;
};

void print_memory(const char *mem, size_t size, std::ostream &os, const PrintOptions &opt = {16, true, true, '.'})
{
    const size_t char_size = 3;
    const size_t hex_scale = 2;
    std::string ascii_buffer;
    const char *end = mem + size;

    if (opt.title)
    {
        const size_t pad = (opt.width * char_size - sizeof(uintptr_t) * hex_scale) / 2;
        os << std::setfill(' ') << std::setw(pad + sizeof(uintptr_t) * hex_scale) << static_cast<const void *>(mem)
           << '\n';
    }

    for (const char *ptr = mem; ptr < end; ++ptr)
    {
        os << std::hex << std::setfill('0') << std::setw(hex_scale) << static_cast<uint16_t>(static_cast<uint8_t>(*ptr))
           << ' ';

        if (opt.ascii)
        {
            ascii_buffer += *ptr >= ' ' && *ptr <= '~' ? *ptr : opt.no_ascii_placeholder;
        }

        if ((ptr - mem) % opt.width == opt.width - 1) // new line
        {
            if (opt.ascii)
            {
                os << " " << ascii_buffer;
                ascii_buffer.clear();
            }

            os << '\n';
        }
    }

    if (opt.ascii)
    {
        os << std::setfill(' ') << std::setw((opt.width - size % opt.width) * char_size + ascii_buffer.size() + 1)
           << ascii_buffer;
    }
}

template<typename T, typename Generator, typename Distribution>
class random_iterator
{
  public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T *;
    using reference = T &;
    using iterator_category = std::forward_iterator_tag;

  public:
    template<typename... Args>
    explicit random_iterator(Args... args) : gen(dev()), distrib(args...), val(distrib(gen)){};

    random_iterator(const random_iterator &x) : gen(x.gen), distrib(x.distrib), val(x.val){};

    random_iterator &operator++()
    {
        val = distrib(gen);
        return *this;
    }

    random_iterator operator++(int)
    {
        auto ret = *this;

        ++*this;

        return ret;
    }

    const T &operator*() { return val; }

  private:
    std::random_device dev;
    Generator gen;
    Distribution distrib;
    T val;
};

using random_int_iterator = random_iterator<int64_t, std::mt19937, std::uniform_int_distribution<int64_t>>;
using random_uint_iterator = random_iterator<uint64_t, std::mt19937, std::uniform_int_distribution<uint64_t>>;
using random_double_iterator = random_iterator<double, std::mt19937, std::uniform_real_distribution<double>>;
/*
 * random_string
 */
template<typename T>
class random_string
{
  public:
    explicit random_string(const T &dict) : dict(dict) {}
    T operator()(size_t length)
    {
        T ret(length, 0);
        random_uint_iterator rnd(0, dict.size() - 1);

        std::generate(ret.begin(), ret.end(), [&] { return dict[*rnd++]; });

        return ret;
    }

  public:
    static random_string hex();
    static random_string digits();
    static random_string alphabet();

  private:
    T dict;
};

template<>
random_string<std::string> random_string<std::string>::hex()
{
    return random_string<std::string>("0123456789abcdef");
};
template<>
random_string<std::wstring> random_string<std::wstring>::hex()
{
    return random_string<std::wstring>(L"0123456789abcdef");
};

template<>
random_string<std::string> random_string<std::string>::digits()
{
    return random_string<std::string>("0123456789");
};
template<>
random_string<std::wstring> random_string<std::wstring>::digits()
{
    return random_string<std::wstring>(L"0123456789");
};

template<>
random_string<std::string> random_string<std::string>::alphabet()
{
    return random_string<std::string>("abcdefghijklmnopqrstuvwxyz");
};
template<>
random_string<std::wstring> random_string<std::wstring>::alphabet()
{
    return random_string<std::wstring>(L"abcdefghijklmnopqrstuvwxyz");
};

/**
 * utf8_iterator class
 */
template<typename Iterator>
class utf8_iterator
{
  public:
    utf8_iterator(Iterator it) : it(it) {}

    utf8_iterator &operator++()
    {
        it += cp_len();
        return *this;
    }

    char32_t operator*()
    {

        uint8_t cp_size = cp_len();
        char32_t cp = 0;

        switch (cp_size)
        {
        case 1:
            cp = *it;
            break;
        case 2:
            cp |= *it & 0x1F;
            break;
        case 3:
            cp |= *it & 0xF;
            break;
        case 4:
            cp |= *it & 0x3;
            break;
        }

        for (auto cp_it = std::next(it); std::distance(it, cp_it) < cp_size; ++cp_it)
        {
            cp <<= 6;
            cp |= *cp_it & 0x3F;
        }

        return cp;
    }

    bool operator!=(Iterator r) const { return it != r; }

  private:
    uint8_t cp_len() const
    {
        if ((*it & 0x80) == 0) // lead bit is zero, must be a single ascii
            return 1;
        else if ((*it & 0xE0) == 0xC0) // 110x xxxx
            return 2;
        else if ((*it & 0xF0) == 0xE0) // 1110 xxxx
            return 3;
        else if ((*it & 0xF8) == 0xF0) // 1111 0xxx
            return 4;
        else
            throw std::runtime_error("Invalid cp");
    }

  private:
    Iterator it;
};

template<typename Container,
         typename T = typename Container::value_type,
         std::enable_if_t<!std::is_base_of_v<std::string, Container>, bool> = true>
std::istream &operator>>(std::istream &is, Container &x)
{
    for (auto it = x.begin(); is.good() && it != x.end(); ++it)
        is >> *it;

    return is;
}
#ifndef PRINT_LENGTH_LIMIT
#define PRINT_LENGTH_LIMIT 15
#endif // !PRINT_LENGTH_LIMIT

template<typename Container,
         typename T = typename Container::value_type,
         std::enable_if_t<!std::is_base_of_v<std::string, Container>, bool> = true>
std::ostream &operator<<(std::ostream &os, const Container &x)
{
    if (x.size() == 0)
    {
        os << "[]";
        return os;
    }

    os << '[';
    auto last = std::prev(x.end());
    if (x.size() > PRINT_LENGTH_LIMIT)
    {
        auto h = std::min<size_t>(x.size(), PRINT_LENGTH_LIMIT) - 1;
        std::copy_n(x.begin(), h, std::ostream_iterator<T>(os, ", "));
        os << "..., ";
    }
    else
    {
        std::copy(x.begin(), last, std::ostream_iterator<T>(os, ", "));
    }
    os << *last << ']';

    return os;
}

template<typename It1, typename It2, typename D>
void merge(It1 first1, It1 last1, It2 first2, It2 last2, D dest)
{
    while (first1 != last1)
    {
        if (first2 == last2)
        {
            std::copy(first1, last1, dest);
            return;
        }

        if (*first1 < *first2)
        {
            *dest++ = *first1++;
        }
        else
        {
            *dest++ = *first2++;
        }
    }

    std::copy(first2, last2, dest);
}

namespace sort
{

template<typename It>
void merge(It first, It last)
{
    auto origin_first = first;
    size_t len = std::distance(first, last);
    std::vector<typename It::value_type> temp(len);
    auto temp_first = temp.begin();
    for (size_t sub_len = 1; sub_len < len; sub_len <<= 1)
    {
        for (size_t n = 0; n < len; n += sub_len << 1)
        {
            auto mid = std::next(first, std::min(n + sub_len, len));

            util::merge(std::next(first, n),
                        mid,
                        mid,
                        std::next(first, std::min(std::distance(first, mid) + sub_len, len)),
                        std::next(temp_first, n));
        }

        std::swap(temp_first, first);
    }

    if (static_cast<int64_t>(std::ceil(std::log2(len))) % 2)
        std::swap_ranges(origin_first, last, temp.begin());
}

} // namespace sort

#ifdef _WIN32

class WinCin
{
    template<typename T>
    friend WinCin &operator>>(WinCin &in, T &x);

  public:
    void Init()
    {
        SetConsoleCP(CP_UTF8);
        cinHandle = GetStdHandle(STD_INPUT_HANDLE);
    }

    bool IsInit() const { return cinHandle != INVALID_HANDLE_VALUE; }

  private:
    void Store(const std::string &str)
    {
        auto pos = stream.tellg();
        stream.seekg(0, std::ios::end);
        std::string temp(stream.tellg() - pos, 0);
        stream.read(temp.data(), temp.size());
        stream = std::stringstream(str + temp);
    }

  private:
    HANDLE cinHandle = INVALID_HANDLE_VALUE;
    std::stringstream stream;
};

template<typename T>
WinCin &operator>>(WinCin &in, T &x)
{
    if (!in.IsInit())
        in.Init();

    if (in.stream >> x)
        return in;

    std::vector<char> buffer(1024);

    DWORD read = 0;
    ReadConsole(in.cinHandle, buffer.data(), static_cast<DWORD>(buffer.size()), &read, NULL);

    DWORD size = WideCharToMultiByte(CP_UTF8, 0, reinterpret_cast<wchar_t *>(buffer.data()), -1, NULL, 0, NULL, NULL);
    std::string utf8(size - 1, 0);

    WideCharToMultiByte(CP_UTF8, 0, reinterpret_cast<wchar_t *>(buffer.data()), -1, utf8.data(), size, NULL, NULL);

    in.Store(utf8);

    in.stream >> x;

    return in;
}

inline WinCin wincin;

#endif // _WIN32
} // namespace util
