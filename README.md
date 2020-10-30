# CppUtil

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
const T& operator*()
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
