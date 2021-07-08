#define TIMING
#include "util.hpp"

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <atomic>
#include <numeric>

void test(bool a)
{
	if (!a)
		throw std::exception();
}

template<typename R, typename T, typename It = typename std::vector<T>::iterator>
R apply(R(*fn)(It, It), std::initializer_list<T> x)
{
	std::vector<T> temp = x;
	return fn(temp.begin(), temp.end());
}

void test_mean()
{
	test(apply<double>(util::mean, { 1, 2, 3, 4, 5 }) == 3);
	test(apply<double>(util::mean, { 1 }) == 1);
}

void test_median()
{
	test(apply<double>(util::median, { 1, 2, 3, 4, 5 }) == 3);
	test(apply<double>(util::median, { 1 }) == 1);
	test(apply<double>(util::median, { 1, 5, 7, 10, 12 }) == 7);
	test(apply<double>(util::median, { 1, 5, 6, 10, 12, 13 }) == 8);
}

void test_variance()
{
	test(apply<double>(util::variance, { 2, 7, 10, 22, 28 }) == 93.76);
	test(apply<double>(util::variance, { 10, 20 }) == 25);
}

void test_standard_deviation()
{
	test(apply<double>(util::standard_deviation, { 10, 20 }) == 5);
}

namespace
{
template<typename It>
bool operator==(const util::max_subarray_result<It>& a, const util::max_subarray_result<It>& b)
{
	return a.begin == b.begin && a.end == b.end && a.sum == b.sum;
}
}

void test_max_subarray_sum()
{
	std::vector<int> a{ 0, 10, 2, 5, -20, 12, 9 };
	util::max_subarray_result<std::vector<int>::iterator> r{
		std::next(a.begin(), 5),
		std::next(a.begin(), 7) ,
		21 };

	test(util::max_subarray_sum(a.begin(), a.end()) == r);
}

void test_matrix()
{
	util::matrix<int64_t> m({ {3, 8}, {4, 6} });

	test(m[0][0] == 3);
	test(m[0][1] == 8);
	test(m[1][0] == 4);
	test(m[1][1] == 6);

	test(m.determinant() == -14);
}

void test_queue()
{
	util::queue<int64_t> q;

	q.push(10);
	test(q.front() == 10);
	test(q.back() == 10);
	q.push(20);
	test(q.front() == 10);
	test(q.back() == 20);
	q.push(30);
	test(q.front() == 10);
	test(q.back() == 30);
	q.push(40);
	test(q.front() == 10);
	test(q.back() == 40);

	int64_t x = 42;
	q.push(x);
	test(q.front() == 10);
	test(q.back() == 42);
	q.pop();
	test(q.front() == 20);
	test(q.back() == 42);
	q.pop();
	q.pop();
	q.pop();
	q.pop();

	test(q.empty());

	try
	{
		q.pop();
		test("Non handled pop on empty queue");
	}
	catch (...) {}

	try
	{
		q.back();
		test("Non handled back on empty queue");
	}
	catch (...) {}
}

void test_fixed_queue()
{
	util::fixed_queue<int64_t, 3> q;

	q.enqueue(1);
	q.enqueue(2);
	q.enqueue(3);

	test(q.enqueue(4) == false);
	test(q.dequeue() == 1);
	test(q.size() == 2);
}

void test_threadsafe_queue()
{
	util::threadsafe_queue<int64_t> q;

	std::vector<std::thread> threadlist;
	const uint64_t threadCount = 10;
	for (uint64_t n = 0; n < threadCount; ++n)
	{
		threadlist.push_back(std::thread([&, n]
		{
			q.push(n);
		}));
	}

	for (auto& t : threadlist)
		t.join();

	test(q.size() == threadCount);

	int64_t sum = 0;

	while (!q.empty())
	{
		sum += q.pop();
	}

	test(sum == (threadCount * (threadCount - 1)) / 2);
}

void test_threadsafe_priority_queue()
{
	util::threadsafe_priority_queue<int64_t> q;

	q.push(10);
	q.push(20);
	q.push(15);

	test(q.pop() == 20);
}

void test_thread_pool()
{
	util::thread_pool p;
	util::task_package t;

	std::atomic<int64_t> x = 0;

	t.append([&] { ++x; });
	t.append([&] { ++x; });
	t.append([&] { ++x; });
	t.append([&] { ++x; });

	p.schedule(t);
	t.wait();

	test(x == 4);
}

void test_parallel_map()
{
	util::thread_pool p;
	std::vector<int64_t> x{ 1, 2, 3, 4, 5 };
	util::parallel_map m(x, p);

	m.map([](int64_t x) { return 2 * x; }).wrun();

	test(x == std::vector<int64_t>{ 2, 4, 6, 8, 10 });
}

void random_iterator_test()
{
	const int64_t max = 10;
	const uint64_t count = 10;
	util::random_int_iterator rnd(0, max);

	std::vector<int64_t> x;
	std::copy_n(rnd, count, std::back_inserter(x));

	test(std::accumulate(x.begin(), x.end(), 0) <= max * count);
}

void test_utf8_iterator()
{
	std::string x = u8"Привет";

	util::utf8_iterator it(x.begin());

	test(*it == *L"П");
	test(*++it == *L"р");
	test(*++it == *L"и");
	test(*++it == *L"в");
	test(*++it == *L"е");
	test(*++it == *L"т");
}

void test_timing()
{
	{
		auto fn = [](auto a, auto b, auto c) { return a * b + c; };
		std::stringstream ss;
		auto result = util::timing("Test#1", ss, fn, 10, 20, 30);

		test(result == 230 && ss.str().find("Test#1") != std::string::npos);
	}

	{
		auto fn = [] {};
		std::stringstream ss;
		util::timing("Test#2", ss, fn);

		test(ss.str().find("Test#2") != std::string::npos);
	}
}

void test_print_memory()
{
	std::string x = "1234567890";
	std::stringstream ss;

	util::print_memory(x.data(), 10, ss, { 16, false, true, '.' });
	auto ll = ss.str();
	test(ss.str() == "31 32 33 34 35 36 37 38 39 30                    1234567890");
}

template<typename T>
void test_merge_impl(std::initializer_list<T>&& c1, std::initializer_list<T>&& c2)
{
	std::vector<T> d1(c1.size() + c2.size());
	std::vector<T> d2(c1.size() + c2.size());

	std::merge(c1.begin(), c1.end(), c2.begin(), c2.end(), d1.begin());
	util::merge(c1.begin(), c1.end(), c2.begin(), c2.end(), d2.begin());

	test(d1 == d2);
}

void test_merge()
{
	test_merge_impl({ 1, 2, 4, 5, 6, 7 }, { 3, 5, 8, 10 });
	test_merge_impl({ 6, 7, 8, 9, 10 }, { 1, 2, 3, 4, 5 });
}

template<typename T>
void test_merge_sort_impl(T&& c)
{
	std::vector<typename std::remove_reference_t<T>::value_type> a1(c.begin(), c.end());
	auto a2 = a1;

	std::sort(a1.begin(), a1.end());
	util::sort::merge(a2.begin(), a2.end());

	test(a1 == a2);
}

void test_merge_sort()
{
	test_merge_sort_impl(std::vector<int64_t>{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
	test_merge_sort_impl(std::vector<int64_t>{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 });
	test_merge_sort_impl(std::vector<int64_t>{ 1, 3, 5, 6, 0, 2, 7, 2 });

	util::random_int_iterator rnd(-10000, 10000);

	std::vector<int64_t> rnd_v(10000);
	std::generate(rnd_v.begin(), rnd_v.end(), [&]() { return *rnd++; });

	test_merge_sort_impl(rnd_v);
}

int main()
{
	try
	{
		test_mean();
		test_median();
		test_variance();
		test_standard_deviation();
		test_max_subarray_sum();
		test_matrix();
		test_queue();
		test_fixed_queue();
		test_threadsafe_queue();
		test_threadsafe_priority_queue();
		test_thread_pool();
		test_parallel_map();
		random_iterator_test();
		test_utf8_iterator();
		test_timing();
		test_print_memory();
		test_merge();
		test_merge_sort();
	}
	catch (...)
	{
		return 1;
	}

	return 0;
}