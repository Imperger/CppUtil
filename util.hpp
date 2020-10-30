#pragma once
#include <iostream>
#include <algorithm>
#include <functional>
#include <optional>
#include <random>
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <condition_variable>
#include <ratio>
#include <vector>
#include <queue>
#include <stdexcept>

namespace util
{
/*
 * threadsafe_queue
 */
template<typename T>
class threadsafe_queue
{
public:
	void push(const T& val)
	{
		std::lock_guard<std::shared_mutex> lk(m);
		queue.push(val);
	}
	T pop()
	{
		std::lock_guard<std::shared_mutex> lk(m);
		T val = queue.front(); queue.pop();
		return val;
	}
	std::optional<T> try_pop()
	{
		std::lock_guard<std::shared_mutex> lk(m);

		if (queue.empty())
			return {};

		T val = queue.front(); queue.pop();
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
/**
 * task_package
 */
class task_package
{
	friend class thread_pool;
public:
	template<typename Fn, typename ...Args>
	void append(Fn fn, Args ...args)
	{
		if (sealed)
			throw std::runtime_error("Append a task to an already scheduled package is forbidden");

		tasks.push_back(std::bind(&task_package::task_wrapper, this, static_cast<typename decltype(tasks)::value_type>(std::bind(fn, args...))));
	}
	void wait()
	{
		std::unique_lock<std::mutex> lk(m);
		complete_event.wait(lk, [&]() { return completed(); });
	}
	bool completed() const
	{
		return completion_counter == tasks.size();
	}
private:
	void task_wrapper(std::function<void()> fn)
	{
		fn();
		++completion_counter;
		complete_event.notify_one();
	}
private:
	std::vector<std::function<void()>> tasks;
	std::atomic<uint64_t> completion_counter;
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
		uint64_t id;
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
			pool[n].t = std::thread(std::bind(&thread_pool::worker_loop, this, std::ref(pool[n])));
		}
	}
	template<typename Fn, typename ...Args>
	void schedule(Fn fn, Args ...args)
	{
		auto candidate = find_free_worker();
		if (candidate)
		{
			worker*& c = *candidate;
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
	void schedule(task_package& p)
	{
		p.sealed = true;

		for (auto t : p.tasks)
			schedule(t);
	}
	bool is_busy()
	{
		return !waitList.empty() || std::any_of(pool.begin(), pool.end(), [](const worker& x) { return (bool)x.is_busy; });
	}
	uint64_t executed_tasks() const
	{
		return task_counter;
	}
	uint64_t size() const
	{
		return pool.size();
	}
	uint64_t queue_length() const
	{
		return waitList.size();
	}
	void release()
	{
		isRunning = false;
		for (worker& x : pool)
		{
			x.waiter.notify_one();
			x.t.join();
		}
	}
	~thread_pool()
	{
		release();
	}
private:
	void worker_loop(worker& self)
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

	std::optional<worker*> find_free_worker()
	{
		auto it = std::find_if(pool.begin(), pool.end(), [&](const worker& x) { return !x.is_busy; });

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
	using transform_type = std::function<value_type(const value_type&)>;
	using iterator = typename Container::iterator;
public:
	explicit parallel_map(Container& target, thread_pool& pool): target(&target), pool(&pool) {}
	template<typename Fn, typename ...Args>
	parallel_map& map(Fn fn, Args ...args)
	{
		transformList.push_back(std::bind(fn, std::placeholders::_1, args...));
		return *this;
	}
	void run()
	{
		auto chunk_size = target->size() / pool->size();
		for (auto begin = target->begin(); begin < target->end(); std::advance(begin, chunk_size))
		{
			auto end = begin + chunk_size;
			if (end > target->end())
				end = target->end();

			pkg.append([this, begin, end]()
				{
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
	Container* target;
	thread_pool* pool;
	task_package pkg;
	std::vector<transform_type> transformList;
};
/**
 * timer class
 */
class timer
{
	using clock = std::chrono::steady_clock;
public:
	void start()
	{
		tm = clock::now();
	}
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

template<typename T, typename Generator, typename Distribution>
class random_iterator
{
public:
	template<typename ...Args>
	explicit random_iterator(Args... args) : gen(dev()), distrib(args...), val(distrib(gen)) {};
	random_iterator(const random_iterator& x) : gen(x.gen), distrib(x.distrib), val(x.val) {};
	random_iterator& operator++()
	{
		val = distrib(gen);
		return *this;
	}
	const T& operator*()
	{
		return val;
	}

private:
	std::random_device dev;
	Generator gen;
	Distribution distrib;
	T val;
};

using random_int_iterator = random_iterator<int, std::mt19937, std::uniform_int_distribution<int>>;
using random_double_iterator = random_iterator<double, std::mt19937, std::uniform_real_distribution<double>>;

template<typename Container, typename T = typename Container::value_type>
std::istream& operator>>(std::istream& is, Container& x)
{
	for (auto it = x.begin(); is.good() && it != x.end(); ++it)
		is >> *it;

	return is;
}

template<typename Container, typename T = typename Container::value_type>
std::ostream& operator<<(std::ostream& os, const Container& x)
{
	if (x.size() == 0)
	{
		os << "[]";
		return os;
	}

	os << '[';
	auto last = std::prev(x.end());
	std::copy(x.begin(), last, std::ostream_iterator<T>(os, ", "));
	os << *last << ']';

	return os;
}
}
