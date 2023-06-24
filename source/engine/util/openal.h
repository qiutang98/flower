#pragma once
#include <AL/al.h>
#include <AL/alc.h>
#include "util.h"

#define alCall(function, ...) alCallImpl(__FILE__, __LINE__, function, __VA_ARGS__)

inline bool check_al_errors(const std::string& filename, const std::uint_fast32_t line)
{
    ALenum error = alGetError();
    if (error != AL_NO_ERROR)
    {
        LOG_ERROR("***ERROR*** ({0}: {1})", filename , line);
        switch (error)
        {
        case AL_INVALID_NAME:
            LOG_ERROR("AL_INVALID_NAME: a bad name (ID) was passed to an OpenAL function");
            break;
        case AL_INVALID_ENUM:
            LOG_ERROR("AL_INVALID_ENUM: an invalid enum value was passed to an OpenAL function");
            break;
        case AL_INVALID_VALUE:
            LOG_ERROR("AL_INVALID_VALUE: an invalid value was passed to an OpenAL function");
            break;
        case AL_INVALID_OPERATION:
            LOG_ERROR("AL_INVALID_OPERATION: the requested operation is not valid");
            break;
        case AL_OUT_OF_MEMORY:
            LOG_ERROR("AL_OUT_OF_MEMORY: the requested operation resulted in OpenAL running out of memory");
            break;
        default:
            LOG_ERROR("UNKNOWN AL ERROR: {}. ", error);
        }
        return false;
    }
    return true;
}

template<typename alFunction, typename... Params>
auto alCallImpl(const char* filename,
    const std::uint_fast32_t line,
    alFunction function,
    Params... params)
    -> typename std::enable_if_t<!std::is_same_v<void, decltype(function(params...))>, decltype(function(params...))>
{
    auto ret = function(std::forward<Params>(params)...);
    check_al_errors(filename, line);
    return ret;
}

template<typename alFunction, typename... Params>
auto alCallImpl(const char* filename,
    const std::uint_fast32_t line,
    alFunction function,
    Params... params)
    -> typename std::enable_if_t<std::is_same_v<void, decltype(function(params...))>, bool>
{
    function(std::forward<Params>(params)...);
    return check_al_errors(filename, line);
}

#define alcCall(function, device, ...) alcCallImpl(__FILE__, __LINE__, function, device, __VA_ARGS__)

inline bool check_alc_errors(const std::string& filename, const std::uint_fast32_t line, ALCdevice* device)
{
    ALCenum error = alcGetError(device);
    if (error != ALC_NO_ERROR)
    {
        LOG_ERROR("***ERROR*** ({0} : {1}).", filename, line);
        switch (error)
        {
        case ALC_INVALID_VALUE:
            LOG_ERROR("ALC_INVALID_VALUE: an invalid value was passed to an OpenAL function");
            break;
        case ALC_INVALID_DEVICE:
            LOG_ERROR("ALC_INVALID_DEVICE: a bad device was passed to an OpenAL function");
            break;
        case ALC_INVALID_CONTEXT:
            LOG_ERROR("ALC_INVALID_CONTEXT: a bad context was passed to an OpenAL function");
            break;
        case ALC_INVALID_ENUM:
            LOG_ERROR("ALC_INVALID_ENUM: an unknown enum value was passed to an OpenAL function");
            break;
        case ALC_OUT_OF_MEMORY:
            LOG_ERROR("ALC_OUT_OF_MEMORY: an unknown enum value was passed to an OpenAL function");
            break;
        default:
            LOG_ERROR("UNKNOWN ALC ERROR: {}.", error);
        }
        return false;
    }
    return true;
}

template<typename alcFunction, typename... Params>
auto alcCallImpl(const char* filename,
    const std::uint_fast32_t line,
    alcFunction function,
    ALCdevice* device,
    Params... params)
    -> typename std::enable_if_t<std::is_same_v<void, decltype(function(params...))>, bool>
{
    function(std::forward<Params>(params)...);
    return check_alc_errors(filename, line, device);
}

template<typename alcFunction, typename ReturnType, typename... Params>
auto alcCallImpl(const char* filename,
    const std::uint_fast32_t line,
    alcFunction function,
    ReturnType& returnValue,
    ALCdevice* device,
    Params... params)
    -> typename std::enable_if_t<!std::is_same_v<void, decltype(function(params...))>, bool>
{
    returnValue = function(std::forward<Params>(params)...);
    return check_alc_errors(filename, line, device);
}

constexpr std::size_t kOpenAlNumBuffers = 4;
constexpr std::size_t kOpenAlBufferSize = 65536; // 32kb of data in each buffer

inline void update_stream(const ALuint source,
    const ALenum& format,
    const std::int32_t& sampleRate,
    const std::vector<char>& soundData,
    std::size_t& cursor)
{
    ALint buffersProcessed = 0;
    alCall(alGetSourcei, source, AL_BUFFERS_PROCESSED, &buffersProcessed);

    if (buffersProcessed <= 0)
        return;

    while (buffersProcessed--)
    {
        ALuint buffer;
        alCall(alSourceUnqueueBuffers, source, 1, &buffer);

        ALsizei dataSize = kOpenAlBufferSize;

        char* data = new char[dataSize];
        std::memset(data, 0, dataSize);

        std::size_t dataSizeToCopy = kOpenAlBufferSize;
        if (cursor + kOpenAlBufferSize > soundData.size())
            dataSizeToCopy = soundData.size() - cursor;

        std::memcpy(&data[0], &soundData[cursor], dataSizeToCopy);
        cursor += dataSizeToCopy;

        if (dataSizeToCopy < kOpenAlBufferSize)
        {
            cursor = 0;
            std::memcpy(&data[dataSizeToCopy], &soundData[cursor], kOpenAlBufferSize - dataSizeToCopy);
            cursor = kOpenAlBufferSize - dataSizeToCopy;
        }

        alCall(alBufferData, buffer, format, data, kOpenAlBufferSize, sampleRate);
        alCall(alSourceQueueBuffers, source, 1, &buffer);

        delete[] data;
    }
}

namespace engine
{

}