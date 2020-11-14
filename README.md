# CppUtil

## matrix

```cpp
template<typename T>
class matrix
```
```cpp
explicit matrix(std::size_t m, std::size_t n)
```
```cpp
template<size_t M, size_t N>
explicit matrix(const T(&mtx)[M][N])
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

## IO overloads for containers

```cpp
template<typename Container, typename T = typename Container::value_type>
std::istream& operator>>(std::istream& is, Container& x)
```

```cpp
template<typename Container, typename T = typename Container::value_type>
std::ostream& operator<<(std::ostream& os, const Container& x)
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
