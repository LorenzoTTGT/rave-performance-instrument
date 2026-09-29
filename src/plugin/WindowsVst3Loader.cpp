// The bundle entry has no LibTorch imports. Load its private implementation only
// after the Windows loader lock has been released, without changing host PATH.
#include <mutex>
#include <string>
#include <windows.h>

namespace
{
HMODULE wrapper = nullptr;
HMODULE implementation = nullptr;
std::mutex moduleMutex;

bool loadImplementation()
{
    if (implementation != nullptr)
        return true;
    std::wstring path(32768, L'\0');
    const auto size = GetModuleFileNameW(wrapper, path.data(), static_cast<DWORD>(path.size()));
    if (size == 0 || size >= path.size())
        return false;
    path.resize(size);
    path = path.substr(0, path.find_last_of(L"\\/")) + L"\\RaveRuntime\\RaveImplementation.dll";
    implementation =
        LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    return implementation != nullptr;
}
} // namespace

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        wrapper = instance;
    }
    return TRUE;
}

extern "C" __declspec(dllexport) bool InitDll()
{
    const std::lock_guard<std::mutex> lock(moduleMutex);
    if (!loadImplementation())
        return false;
    const auto initialise = reinterpret_cast<bool (*)()>(GetProcAddress(implementation, "InitDll"));
    if (initialise != nullptr && initialise())
        return true;
    FreeLibrary(implementation);
    implementation = nullptr;
    return false;
}

extern "C" __declspec(dllexport) void *GetPluginFactory()
{
    const std::lock_guard<std::mutex> lock(moduleMutex);
    if (!loadImplementation())
        return nullptr;
    const auto factory = reinterpret_cast<void *(*)()>(GetProcAddress(implementation, "GetPluginFactory"));
    return factory != nullptr ? factory() : nullptr;
}

extern "C" __declspec(dllexport) bool ExitDll()
{
    const std::lock_guard<std::mutex> lock(moduleMutex);
    if (implementation == nullptr)
        return true;
    const auto finish = reinterpret_cast<bool (*)()>(GetProcAddress(implementation, "ExitDll"));
    const bool result = finish != nullptr && finish();
    FreeLibrary(implementation);
    implementation = nullptr;
    return result;
}
