/********************************************************************
file base:      Logger.h
author:         LZD
created:        2025/08/07
purpose:        日志文件系统
*********************************************************************/
#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <ctime>
#include <iomanip>
#include <thread>
#include <string>

namespace LFMVS
{
	enum LogLevel
		{
			LOG_DEBUG,
			LOG_INFO,
			LOG_WARN,
			LOG_ERROR
		};

	class Logger
		{
		public:
			static Logger& instance();

			void setLogDir(const std::string& directory);
			void setLevel(LogLevel level);
			LogLevel getLevel();
			void logImpl(LogLevel level, const std::string& msg);  // 实际执行日志写入

			template<typename... Args>
			void log(LogLevel level, Args&&... args);  // 不再调用 log(...) 本体

		private:
			Logger();
			std::string levelToStr(LogLevel level);
			std::string currentTime();
			std::string currentDate();
			void rotateLogIfNeeded();
			void createLogDirIfNeeded();

			std::ofstream logfile_;
			std::mutex mutex_;
			LogLevel current_level_ = LOG_DEBUG;
			std::string log_directory_ = ".";
			std::string current_log_date_;

			// append 重载
			void append(std::ostringstream& oss, const char* value);
			template<int N>
			void append(std::ostringstream& oss, const char(&value)[N]);

			template<typename T>
			void append(std::ostringstream& oss, const T& value);

			template<typename T, typename... Args>
			void append(std::ostringstream& oss, const T& value, Args&&... args);
		};

		// 宏定义
		#define LOG_DEBUG(...) Logger::instance().log(LOG_DEBUG, __VA_ARGS__)
		#define LOG_INFO(...)  Logger::instance().log(LOG_INFO,  __VA_ARGS__)
		#define LOG_WARN(...)  Logger::instance().log(LOG_WARN,  __VA_ARGS__)
		#define LOG_ERROR(...) Logger::instance().log(LOG_ERROR, __VA_ARGS__)

		// append 实现
		inline void Logger::append(std::ostringstream& oss, const char* value) {
			oss << value;
		}

		template<int N>
		inline void Logger::append(std::ostringstream& oss, const char(&value)[N]) {
			oss << value;
		}

		template<typename T>
		inline void Logger::append(std::ostringstream& oss, const T& value) {
			oss << value;
		}

		template<typename T, typename... Args>
		inline void Logger::append(std::ostringstream& oss, const T& value, Args&&... args) {
			append(oss, value);
			append(oss, std::forward<Args>(args)...);
		}

		template<typename... Args>
		inline void Logger::log(LogLevel level, Args&&... args) {
			std::ostringstream oss;
			append(oss, std::forward<Args>(args)...);
			logImpl(level, oss.str());  // ❗改为 logImpl，避免递归 log()
		}
}
#endif //LOGGER_H
