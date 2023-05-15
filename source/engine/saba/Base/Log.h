//
// Copyright(c) 2016-2017 benikabocha.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)
//

#ifndef SABA_BASE_LOG_H_
#define SABA_BASE_LOG_H_

#include <util/log.h>

#define SABA_INFO LOG_INFO

#define SABA_WARN LOG_WARN

#define SABA_ERROR LOG_ERROR

#define SABA_ASSERT(expr)\
	assert(expr)

#endif // !SABA_BASE_LOG_H_

