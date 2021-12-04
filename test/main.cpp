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
	{
		std::vector<int> a;

		test(util::max_subarray_sum(a.begin(), a.end()) ==
			util::max_subarray_result<std::vector<int>::iterator>{a.begin(), a.end(), 0});
	}
	{
		std::vector<int> a{ 0, 10, 2, 5, -20, 12, 9 };
		util::max_subarray_result<std::vector<int>::iterator> r{
			std::next(a.begin(), 5),
			std::next(a.begin(), 7) ,
			21 };

		test(util::max_subarray_sum(a.begin(), a.end()) == r);
	}
}

void test_matrix()
{
	{
		util::matrix<int64_t> m({ { 1, 2 } });

		try
		{
			m.determinant();
			test(false);
		}
		catch (std::runtime_error const&)
		{
			test(true);
		}
	}
	{
		util::matrix<int64_t> m({ { 1 } });

		test(m.determinant() == m[0][0]);
	}
	{
		util::matrix<int64_t> m({ {3, 8}, {4, 6} });

		test(m[0][0] == 3);
		test(m[0][1] == 8);
		test(m[1][0] == 4);
		test(m[1][1] == 6);

		test(m.determinant() == -14);
	}
	{
		util::matrix<int64_t> m({ { 6, 1, 1 }, { 4, -2, 5 }, { 2, 8, 7 } });

		test(m.determinant() == -306);
	}
}

void test_longest_common_subsequence()
{
	struct testcase
	{
		std::string a;
		std::string b;
		std::vector<std::string::const_iterator> expect;
	};

	testcase t0{ "", "" };
	t0.expect = { };

	testcase t1{ "aava", "aaa" };
	t1.expect = { t1.a.begin(), t1.a.begin() + 1, t1.a.begin() + 3 };

	testcase t2{ "aaa", "aava" };
	t2.expect = { t2.a.begin(), t2.a.begin() + 1, t2.a.begin() + 2 };

	testcase t3{ "abcdefg", "bcfg" };
	t3.expect = { t3.a.begin() + 1,
		t3.a.begin() + 2,
		t3.a.begin() + 5,
		t3.a.begin() + 6 };

	std::vector<testcase const*> testcases{ &t0, &t1, &t2, &t3 };

	for (auto const* t : testcases)
	{
		auto ret = util::longest_common_subsequence(t->a.begin(), t->a.end(), t->b.begin(), t->b.end());
		test(ret == t->expect);
	}
}

void test_longest_common_substring()
{
	struct testcase
	{
		std::string a;
		std::string b;
		std::pair<std::string::const_iterator, std::string::const_iterator> expect;
	};

	testcase t1{ "aava", "aaa" };
	t1.expect = std::make_pair(t1.a.begin(), t1.a.begin() + 2);

	testcase t2{ "aaa", "aava" };
	t2.expect = std::make_pair(t2.a.begin(), t2.a.begin() + 2);

	testcase t3{ "abcdef", "bcdf" };
	t3.expect = std::make_pair(t3.a.begin() + 1, t3.a.begin() + 4);

	// 'bc' and 'fg' has same length but search starts from the end and 'fg' being found first
	testcase t4{ "abcdefg", "bcfg" };
	t4.expect = std::make_pair(t4.a.begin() + 5, t4.a.begin() + 7);

	std::vector<testcase const*> testcases{ &t1 };

	for (auto const* t : testcases)
	{
		auto ret = util::longest_common_substring(t->a.begin(), t->a.end(), t->b.begin(), t->b.end());
		test(ret == t->expect);
	}
}

void test_longest_palindrome_subsequence()
{
	std::string t = "abcdxjdocboa";
	std::vector<std::string::const_iterator> expect{ t.begin(),
		t.begin() + 1,
		t.begin() + 2,
		t.begin() + 3,
		t.begin() + 5,
		t.begin() + 6,
		t.begin() + 8,
		t.begin() + 9,
		t.begin() + 11 };

	test(util::longest_palindrome_subsequence(t.cbegin(), t.cend()) == expect);
}

void test_queue()
{
	util::queue<int64_t> q;

	try
	{
		q.front();
		test(false);
	}
	catch (std::runtime_error const& e)
	{
		test(true);
	}

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

	try
	{
		q.dequeue();
		test(false);
	}
	catch (std::runtime_error const& e)
	{
		test(true);
	}

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

	{
		util::thread_pool p;
		util::task_package t;

		t.append([]() {});
		p.schedule(t);

		try
		{
			t.append([]() {});
			test(false);
		}
		catch (std::runtime_error const& e)
		{
			test(true);
		}

		t.wait();
	}
}

void test_ms_to_string()
{
	test(util::ms_to_string(750, 2) == "750.00ms");
	test(util::ms_to_string(6000, 2) == "6.00s");
	test(util::ms_to_string(15 * 60 * 1000, 2) == "15.00m");
	test(util::ms_to_string(7 * 60 * 60 * 1000, 2) == "7.00h");
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
	{

		const int64_t max = 10;
		const uint64_t count = 10;
		util::random_int_iterator rnd(0, max);

		std::vector<int64_t> x;
		std::copy_n(rnd, count, std::back_inserter(x));

		test(std::accumulate(x.begin(), x.end(), 0) <= max * count);
	}
	{
		util::random_int_iterator rnd(0, 100);

		auto r0 = *rnd;
		auto r1 = *rnd++;

		test(r0 == r1);
	}
}

template<typename Gen, typename Pred>
void random_string_test(Gen&& gen, Pred&& pred)
{
	{
		test(gen(0).empty());

		for (size_t length = 1; length <= 512; length <<= 1) {

			for (size_t n_test = 0; n_test < 10; ++n_test)
			{
				auto str = gen(length);
				test(str.size() == length && std::all_of(str.begin(), str.end(), pred));
			}
		}
	}
	return;
}

void test_utf8_iterator()
{
	{
		std::u8string x = u8"abc";

		util::utf8_iterator it(x.begin());

		test(*it == L'a');
		test(*++it == L'b');
		test(*++it == L'c');
	}
	{

		std::u8string x = u8"Привет";

		util::utf8_iterator it(x.begin());

		test(*it == L'П');
		test(*++it == L'р');
		test(*++it == L'и');
		test(*++it == L'в');
		test(*++it == L'е');
		test(*++it == L'т');
	}
	{
		std::u8string x = u8"诶比西";

		util::utf8_iterator it(x.begin());

		test(*it == L'诶');
		test(*++it == L'比');
		test(*++it == L'西');
	}
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
	std::string x = "1234567890abcdefghijklmnopqrstuvwxyz";
	std::stringstream ss;
	const util::PrintOptions options = { 16, true, true, '.' };

	util::print_memory(x.data(), x.size(), ss, options);
	std::stringstream expect;
	const size_t char_size = 3;
	const size_t hex_scale = 2;
	const size_t pad = (options.width * char_size - sizeof(uintptr_t) * hex_scale) / 2;

	expect << std::setfill(' ') << std::setw(pad + sizeof(uintptr_t) * hex_scale) << static_cast<const void*>(x.data()) << '\n';
	expect << "31 32 33 34 35 36 37 38 39 30 61 62 63 64 65 66  1234567890abcdef\n"
		"67 68 69 6a 6b 6c 6d 6e 6f 70 71 72 73 74 75 76  ghijklmnopqrstuv\n"
		"77 78 79 7a                                      wxyz";

	test(ss.str() == expect.str());
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

void test_levenshtein_distance()
{
	{
		std::string a = "Hello";
		std::string b = "Hi";

		auto result = util::levenshtein_distance(
			a.begin(), a.end(),
			b.begin(), b.end());

		test(result == 4);
	}

	{
		std::string a = "Hello";
		std::string b = "Hll";

		auto result = util::levenshtein_distance(
			a.begin(), a.end(),
			b.begin(), b.end());

		test(result == 2);
	}

	{
		std::string a = "Hello";
		std::string b = "Hello";

		auto result = util::levenshtein_distance(
			a.begin(), a.end(),
			b.begin(), b.end());

		test(result == 0);
	}

	{
		std::string a = "Hello";
		std::string b = "Hi";

		auto result = util::levenshtein_distance(
			a.begin(), a.end(),
			b.begin(), b.end(), { 2, 2, 1 });

		test(result == 7);
	}
}

#ifndef __clang__
void test_literals()
{
	using namespace util::distance_literals;

	test(1_km == 100000_cm);
	test(1_m + 1_m == 2_m);
	test(1_m + 1_m == 200_cm);
	test(1_m + 2_cm == 102_cm);
	test(1_m - 100_cm == 0_km);
	test(1_m - 10_cm == 90_cm);
	test(1_cm - 1_m == -99_cm);
	test(10_mm == 1_cm);
	test(1_mile < 2_km);
	test(1_m != 1_cm);

	test(1._km == 100000._cm);
	test(1._m + 1._m == 2._m);
	test(1._m + 1._m == 200._cm);
	test(1._m + 2._cm == 102._cm);
	test(1._m - 100._cm == 0._km);
	test(1._m - 10._cm == 90._cm);
	test(1._cm - 1._m == -99._cm);
	test(10._mm == 1._cm);
	test(1._mile <= 1.7_km);
	test(1._m != 1._cm);
}
#endif // __clang__

template<typename T>
void test_lomuto_partition_impl(std::vector<T> const& c, size_t pivot, size_t expected)
{
	auto t = c;
	auto it = util::lomuto_partition(t.begin(), t.end(), t.begin() + pivot);

	test(it == t.begin() + expected);
}

void test_lomuto_partition()
{
	test_lomuto_partition_impl<int>({ 2, 4, 7, 1, 0, 3 }, 5, 3);
	test_lomuto_partition_impl<int>({ 5, 4, 3, 2, 1 }, 0, 4);
	test_lomuto_partition_impl<int>({ 5, 4, 3, 2, 1 }, 4, 0);
	test_lomuto_partition_impl<int>({ 2, 1, 3, 5, 4 }, 2, 2);
	test_lomuto_partition_impl<int>({ 2, 1, 4, 3, 8, 6 }, 2, 3);
}

template<typename T>
void test_quick_select_impl(std::vector<T> const& c, size_t k, T expected)
{
	auto t = c;
	util::quick_select(t.begin(), t.end(), k);

	test(*(t.begin() + k) == expected);
}

void test_quick_select()
{
	test_quick_select_impl<int>({ 1, 2, 3, 4, 5 }, 2, 3);
	test_quick_select_impl<int>({ 5, 4, 3, 2, 1 }, 2, 3);
	test_quick_select_impl<int>({ 2, 1, 4, 3, 8, 6 }, 4, 6);
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
		test_longest_common_subsequence();
		test_longest_common_substring();
		test_longest_palindrome_subsequence();
		test_queue();
		test_fixed_queue();
		test_threadsafe_queue();
		test_threadsafe_priority_queue();
		test_thread_pool();
		test_ms_to_string();
		test_parallel_map();
		random_iterator_test();
		random_string_test(util::random_string<std::string>::alphabet(), [](char x) {return x >= 'a' && x <= 'z'; });
		random_string_test(util::random_string<std::wstring>::alphabet(), [](char x) {return x >= 'a' && x <= 'z'; });
		random_string_test(util::random_string<std::string>::digits(), [](char x) {return x >= '0' && x <= '9'; });
		random_string_test(util::random_string<std::wstring>::digits(), [](char x) {return x >= '0' && x <= '9'; });
		random_string_test(util::random_string<std::string>::hex(), [](char x) {return (x >= '0' && x <= '9') || (x >= 'a' && x <= 'f'); });
		random_string_test(util::random_string<std::wstring>::hex(), [](char x) {return x >= ('0' && x <= '9') || (x >= 'a' && x <= 'f'); });
		test_utf8_iterator();
		test_timing();
		test_print_memory();
		test_merge();
		test_merge_sort();
		test_levenshtein_distance();
		test_lomuto_partition();
		test_quick_select();
#ifndef __clang__
		test_literals();
#endif // __clang__
	}
	catch (...)
	{
		return 1;
	}

	return 0;
}
