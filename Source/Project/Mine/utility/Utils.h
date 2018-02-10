#ifndef __UTILS_H_
#define __UTILS_H_

#include <memory>

namespace mn {

	using uint = unsigned int;
	using uchar = unsigned char;
	using uint64 = unsigned long long int;
	using MCSize = uint;

	template<typename T>
	 T min(T a, T b) { return a <= b ? a : b; }
	template<typename T>
	 T max(T a, T b) { return a > b ? a : b; }

	 template<typename ... Args>
	 std::string string_format(const char* format, Args ... args) {
		 size_t size = snprintf(nullptr, 0, format, args ...) + 1; // Extra space for '\0'
		 std::unique_ptr<char[]> buf(new char[size]);
		 snprintf(buf.get(), size, format, args ...);
		 return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
	 }
	
	template<typename ... Args>
	std::string string_format(const std::string&& format, Args ... args) {
		size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
		std::unique_ptr<char[]> buf(new char[size]);
		snprintf(buf.get(), size, format.c_str(), args ...);
		return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
	}

	/*
	template<typename... Args>
	auto make_vector(Args&&... args) {
		using Item = std::common_type_t<Args...>;
		std::vector<Item>	result(sizeof...(Args));
		// works as a building block
		forArgs(
			[&result](auto&& x) {result.emplace_back(std::forward<decltype(x)>(x)); },
			std::forward<Args>(args)...
		);
		return result;
	}
	*/
}

#endif