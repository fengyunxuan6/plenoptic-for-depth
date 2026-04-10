/********************************************************************
file base:      Logger.cpp
author:         LZD
created:        2025/08/07
purpose:        日志文件系统
*********************************************************************/
#include "Logger.h"
#include <sys/stat.h>
#include <iostream>

#ifdef _WIN32
#include <direct.h>
#endif

namespace LFMVS
{
	Logger& Logger::instance()
	{
        // 使用局部静态变量方式实现线程安全的单例模式
        static Logger instance;
        return instance;
	}

    Logger::Logger() {
        // 延迟初始化目录与文件，避免 log_directory_ 尚未设置
    }

	void Logger::setLogDir(const std::string& directory) {
		log_directory_ = directory;
	}

	void Logger::setLevel(LogLevel level) {
		current_level_ = level;
	}

	LogLevel Logger::getLevel()
	{
		return current_level_;
	}

	void Logger::logImpl(LogLevel level, const std::string& msg) {
        // 添加保护，防止在程序退出时访问已析构的对象
        try {
            if (level < current_level_) return;

            std::lock_guard<std::mutex> lock(mutex_);

            // 第一次写入时才初始化日志文件
            if (!logfile_.is_open()) {
                createLogDirIfNeeded();
                current_log_date_ = currentDate();
                std::string filename = log_directory_ + "/log_" + current_log_date_ + ".txt";
                logfile_.open(filename, std::ios::app);
                if (!logfile_.is_open()) {
                    std::cerr << "❌ 无法打开日志文件: " << filename << std::endl;
                    return;
                }
            }

            rotateLogIfNeeded();

            std::string timeStr = currentTime();
            std::string levelStr = levelToStr(level);
            std::string logLine = "[" + timeStr + "] [" + levelStr + "] " + msg;

            // 输出到控制台
            if (level >= LOG_ERROR)
            {
                std::cout << logLine << std::endl;
            }

            // 写入文件
            if (logfile_.is_open()) {
                logfile_ << logLine << std::endl;
                logfile_.flush();  // 立即刷新到磁盘
            }
        } catch (...) {
            // 忽略日志记录期间发生的任何异常
            // 特别是在程序退出时可能发生的mutex或文件流相关异常
        }
	}

	std::string Logger::levelToStr(LogLevel level) {
		switch (level) {
		case LOG_DEBUG: return "DEBUG";
		case LOG_INFO:  return "INFO";
		case LOG_WARN:  return "WARN";
		case LOG_ERROR: return "ERROR";
		default:        return "UNKNOWN";
		}
	}

	std::string Logger::currentTime() {
		std::time_t now = std::time(nullptr);
		std::tm local_tm;
	#ifdef _WIN32
		localtime_s(&local_tm, &now);
	#else
		local_tm = *std::localtime(&now);
	#endif
		std::ostringstream oss;
		oss << std::put_time(&local_tm, "%H:%M:%S");
		return oss.str();
	}

	std::string Logger::currentDate() {
		std::time_t now = std::time(nullptr);
		std::tm local_tm;
	#ifdef _WIN32
		localtime_s(&local_tm, &now);
	#else
		local_tm = *std::localtime(&now);
	#endif
		std::ostringstream oss;
		oss << std::put_time(&local_tm, "%Y-%m-%d");
		return oss.str();
	}

	void Logger::rotateLogIfNeeded() {
		std::string today = currentDate();
		if (today != current_log_date_) {
			logfile_.close();
			current_log_date_ = today;
			std::string filename = log_directory_ + "/log_" + current_log_date_ + ".txt";
			logfile_.open(filename, std::ios::app);
		}
	}

	void Logger::createLogDirIfNeeded() {
	#ifdef _WIN32
		_mkdir(log_directory_.c_str());
	#else
		mkdir(log_directory_.c_str(), 0755);
	#endif
	}
}