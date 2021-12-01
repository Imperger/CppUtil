# CppUtil

[![codecov](https://codecov.io/gh/Imperger/CppUtil/branch/main/graph/badge.svg?token=LSB7QWSZ41)](https://codecov.io/gh/Imperger/CppUtil)
## mean
```cpp
template<typename It>
inline double mean(It begin, It end)
```

## median

```cpp
template<typename It>
double median(It begin, It end)
```

## variance

```cpp
template<typename It>
inline double variance(It begin, It end)
```

## standard_deviation

```cpp
template<typename It>
inline double standard_deviation(It begin, It end)
```

## max_subarray_sum

```cpp
template<typename It>
max_subarray_result<It> max_subarray_sum(It begin, It end)
```
```cpp
template<typename It>
struct max_subarray_result
{
	It begin;
	It end;
	typename It::value_type sum;
};
```

## matrix

```cpp
template<typename T>
class matrix
```
```cpp
explicit matrix(std::size_t m, std::size_t n)
```
```cpp
template<typename Type, size_t M, size_t N>
explicit matrix(const Type(&mtx)[M][N]): m(M), n(N)
```
```cpp
row_holder operator[](std::size_t idx)
```
```cpp
const_row_holder operator[](std::size_t idx) const
```
```cpp
T determinant() const
```

## Longest common subsequence

```cpp
template<typename RndIt1, typename RndIt2>
std::vector<RndIt1> longest_common_subsequence(RndIt1 first1, RndIt1 last1, RndIt2 first2, RndIt2 last2)
```

## Longest common substring

```cpp
template<typename RndIt1, typename RndIt2>
std::pair<RndIt1> longest_common_substring(RndIt1 first1, RndIt1 last1, RndIt2 first2, RndIt2 last2)
```

## Longest palindrome subsequence

```cpp
template<typename RndIt>
std::vector<RndIt> longest_palindrome_subsequence(RndIt first, RndIt last)
```

## thread_pool

```cpp
class thread_pool
```
---
```cpp
explicit thread_pool(size_t t = std::thread::hardware_concurrency())
```

```cpp
template<typename Fn, typename ...Args>
void schedule(Fn fn, Args ...args)
```
 
```cpp
 void schedule(task_package& p)
```
 
```cpp
bool is_busy()
```
 
```cpp
uint64_t executed_tasks() const
```
```cpp
uint64_t size() const
```
```cpp
uint64_t queue_length() const
```
```cpp
void release()
```
```cpp
void join()
```
```cpp
~thread_pool()
```
## parallel_map

```cpp
template<typename Container>
class parallel_map
```
---
```cpp
explicit parallel_map(Container& target, thread_pool& pool): target(&target), pool(&pool) {}
```
```cpp
template<typename Fn, typename ...Args>
parallel_map& map(Fn fn, Args ...args)
```
```cpp
void run()
```
```cpp
void wrun()
```
## task_package

```cpp
class task_package
```
---
```cpp
template<typename Fn, typename ...Args>
void append(Fn fn, Args ...args)
```
```cpp
void wait()
```
```cpp
bool completed() const
```

## timer

```cpp
class timer
```
---
```cpp
void start()
```
```cpp
uint64_t stop()
```
```cpp
uint64_t reset()
```
## formaters

```cpp
template<typename Fn, 
		 typename... Args, 
		 typename = std::enable_if_t<has_return_type<Fn, Args...>::value>>
auto timing(const char *msg, std::ostream &os, Fn &&fn, Args &&... args)
```
```cpp
template<typename Fn,
         typename... Args,
         typename = std::enable_if_t<std::negation_v<std::bool_constant<has_return_type<Fn, Args...>::value>>>>
void timing(const char *msg, std::ostream &os, Fn &&fn, Args &&... args)
```
### print memory

```cpp
void print_memory(const char* mem, size_t size, std::ostream& os, const PrintOptions& opt = { 16, true, true, '.' })
```
```cpp
struct PrintOptions
{
	size_t width;
	bool title;
	bool ascii;
	char no_ascii_placeholder;
};
```

## random_iterator

```cpp
class random_iterator
```
---
```cpp
template<typename ...Args>
explicit random_iterator(Args... args)
```
```cpp
random_iterator& operator++()
```
```cpp
random_iterator operator++(int)
```
```cpp
const T& operator*()
```

## random_string
```cpp
template<typename T>
class random_string
```
---
```cpp
explicit random_string(const T& dict)
```
```cpp
T operator()(size_t length)
```
```cpp
static random_string hex();
```
```cpp
static random_string digits();
```
```cpp
static random_string alphabet();
```

## utf8_iterator
```cpp
template<typename Iterator>
class utf8_iterator
```
---
```cpp
utf8_iterator(Iterator it) : it(it) {}
```
```cpp
utf8_iterator& operator++()
```
```cpp
char32_t operator*()
```
```cpp
bool operator!=(Iterator r) const
```

## levenshtein_distance
```cpp
template<typename It>
uint64_t levenshtein_distance(It first1, It last1, It first2, It last2, levenshtein_costs const &costs = {1, 1, 1})
```
## IO overloads for containers

```cpp
template<typename Container, typename T = typename Container::value_type>
std::istream& operator>>(std::istream& is, Container& x)
```

```cpp
template<typename Container, typename T = typename Container::value_type>
std::ostream& operator<<(std::ostream& os, const Container& x)
```

## queue

```cpp
template<typename T>
class queue
```
```cpp
template<typename V>
void push(V &&val)
```
```cpp
void pop()
```
```cpp
T const &front() const
```
```cpp
T &front()
 ```
 ```cpp
 T &back()
 ```
 ```cpp
 T const &back() const
 ```
 ```cpp
 bool empty() const
 ```

## fixed_queue

```cpp
template<typename T, size_t N>
class fixed_queue
```
```cpp
bool enqueue(const T& value)
```
```cpp
T dequeue()
```
```cpp
inline bool empty() const
```
```cpp
inline bool full() const
```
```cpp
inline size_t capacity() const
```
```cpp
inline size_t size() const
```


## threadsafe_queue

```cpp 
template<typename T>
class threadsafe_queue
```
---
```cpp 
void push(const T& val)
```
```cpp 
T pop()
```
```cpp 
std::optional<T> try_pop()
```
```cpp 
bool empty() const
```
```cpp 
size_t size() const
```

## threadsafe_priority_queue

```cpp
template<typename T>
class threadsafe_priority_queue
```
---
```cpp
void push(const T& val)
```
```cpp
T pop()
```
```cpp
std::optional<T> try_pop()
```
```cpp
T& top()
```
```cpp
bool empty() const
```
```cpp
size_t size() const
```

## merge

```cpp
template<typename It1, typename It2, typename D>
void merge(It1 first1, It1 last1, It2 first2, It2 last2, D dest)
```

## sort::merge
```cpp
template<typename It>
void merge(It first, It last)
```

## lomuto_partition
```cpp
template<typename It>
It lomuto_partition(It first, It second, It pivot)
```

## quick_select
```cpp
template<typename It, typename P = std::function<It(It, It, It)>>
void quick_select(It first, It second, size_t k, P partition = &lomuto_partition<It>)
```

## literals
```cpp
template<typename T, typename Unit>
class prefixed_value
```

```cpp
template<typename To, typename T, typename U>
constexpr To prefixed_value_cast(prefixed_value<T, U> const &value)
```

```cpp
template<typename T1, typename U1, typename T2, typename U2>
constexpr bool operator==(prefixed_value<T1, U1> const &l, prefixed_value<T2, U2> const &r)
```

```cpp
template<typename T1, typename U1, typename T2, typename U2>
constexpr bool operator!=(prefixed_value<T1, U1> const &l, prefixed_value<T2, U2> const &r)
```

```cpp
template<typename T1, typename U1, typename T2, typename U2>
constexpr std::common_type_t<prefixed_value<T1, U1>, prefixed_value<T2, U2>> operator+(prefixed_value<T1, U1> const &l,
                                                                                       prefixed_value<T2, U2> const &r)
```

```cpp
template<typename T1, typename U1, typename T2, typename U2>
constexpr std::common_type_t<prefixed_value<T1, U1>, prefixed_value<T2, U2>> operator-(prefixed_value<T1, U1> const &l,
                                                                                       prefixed_value<T2, U2> const &r)
```

```cpp
template<typename T1, typename U1, typename T2, typename U2>
constexpr std::common_type_t<prefixed_value<T1, U1>, prefixed_value<T2, U2>> operator*(prefixed_value<T1, U1> const &l,
                                                                                       prefixed_value<T2, U2> const &r)
```

```cpp
template<typename T1, typename U1, typename T2, typename U2>
constexpr std::common_type_t<prefixed_value<T1, U1>, prefixed_value<T2, U2>> operator/(prefixed_value<T1, U1> const &l,
                                                                                       prefixed_value<T2, U2> const &r)
```

## distance_literals

```cpp
template<typename T, typename Unit>
class distance : public literals::prefixed_value<T, Unit>
```

```cpp
template<typename T>
using km = distance<T, literals::unit<1000.l>>
```

```cpp
template<typename T>
using m = distance<T, literals::unit<1.l>>
```

```cpp
template<typename T>
using cm = distance<T, literals::unit<0.01l>>
```

```cpp
template<typename T>
using mm = distance<T, literals::unit<0.001l>>
```

```cpp
template<typename T>
using mile = distance<T, literals::unit<1609.344l>>
```

```cpp
km<int64_t> operator""_km(unsigned long long x)
```

```cpp
m<int64_t> operator""_m(unsigned long long x)
```

```cpp
cm<int64_t> operator""_cm(unsigned long long x)
```

```cpp
mm<int64_t> operator""_mm(unsigned long long x)
```

```cpp
mile<int64_t> operator""_mile(unsigned long long x)
```

```cpp
km<long double> operator""_km(long double x)
```

```cpp
m<long double> operator""_m(long double x)
```

```cpp
cm<long double> operator""_cm(long double x)
```

```cpp
mm<long double> operator""_mm(long double x)
```

```cpp
mile<long double> operator""_mile(long double x)
```
