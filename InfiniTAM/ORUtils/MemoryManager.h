//
// Created by drinkingcoder on 17-11-15.
//

#ifndef INFINITAM_MEMORYMANAGER_H
#define INFINITAM_MEMORYMANAGER_H

#include <cstdlib>
#include <cstring>
#include "MemoryDeviceType.h"
#include "PlatformIndependence.h"

#include "CUDADefines.h"

namespace ORUtils {

    /*!
     * @brief This class is a general data container for both cpu and gpu.
     * @class MemoryManager
     * @tparam T
     */
    /// declartion
    template<typename T>
    class  MemoryManager {
    public:
        enum MemoryCopyDirection {
            CPU_TO_CPU,
            CPU_TO_GPU,
            GPU_TO_CPU,
            GPU_TO_GPU
        };

        /// basic class methods
        explicit MemoryManager(size_t data_size, bool allocate_cpu, bool allocate_gpu):
                m_is_allocated_cpu(false),
                m_is_allocated_gpu(false) {
            Allocate(data_size, allocate_cpu, allocate_gpu);
            Clear();
        }

        explicit MemoryManager(size_t data_size, MemoryDeviceType memory_type):
                m_is_allocated_cpu(false),
                m_is_allocated_gpu(false) {
            switch(memory_type) {
                case MEMORYDEVICE_CPU:
                    Allocate(data_size, true, false);
                    break;
                case MEMORYDEVICE_CUDA:
                    Allocate(data_size, false, true);
                    break;
            }
            InitData();
        }

        virtual ~MemoryBlock() {
            Free();
        }

        /// abandon copy constructor and assignment
        MemoryManager(const MemoryManager&) = delete;
        MemoryManager& operator=(const MemoryManager&) = delete;

        /// basic container method
        virtual void Allocate(size_t data_size, bool allocate_cpu, bool allocate_gpu);
        virtual void Free();
        virtual void InitData(unsigned char default_value = 0);
        virtual void Swap(MemoryManager<T>& rhs);
        virtual void Resize(size_t data_size, bool reallocation = true);
        virtual void SetData(const MemoryManager<T> *source, MemoryCopyDirection copy_direction);
        virtual T GetElement(int idx, MemoryDeviceType memory_type) const; ///< get an individual element from cpu or gpu.

        /// data synchronization between cpu and gpu
        virtual void CopyDataFromDeviceToHost() const;
        virtual void CopyDataFromHostToDevice() const;

        /// property retrieval
        virtual size_t GetDataSize() {
            return m_data_size;
        }

        virtual T* GetData(MemoryDeviceType memory_type);
        virtual const T* GetData(MemoryDeviceType memory_type) const;


    protected:
        bool m_is_allocated_cpu, m_is_allocated_gpu;
        T *m_data_cpu;
        T *m_data_gpu;
        size_t m_data_size;
    private:
    };

    /// implementation

    template <typename T>
    T* MemoryManager<T>::GetData(MemoryDeviceType memory_type) {
        switch (memory_type) {
            case MEMORYDEVICE_CPU: return m_data_cpu;
            case MEMORYDEVICE_CUDA: return m_data_gpu;
        }
        return 0;
    }

    template <typename T>
    const T* MemoryManager<T>::GetData(MemoryDeviceType memory_type) const {
        switch (memory_type) {
            case MEMORYDEVICE_CPU: return m_data_cpu;
            case MEMORYDEVICE_CUDA: return m_data_gpu;
        }
        return 0;
    }

    template <typename T>
    void MemoryManager<T>::Allocate(size_t data_size, bool allocate_cpu, bool allocate_gpu) {
        Free();

        m_data_size = data_size;
        if(allocate_cpu) {
            if(data_size == 0) {
                m_data_cpu = NULL;
            } else {
               m_data_cpu = new T[m_data_size];
            }
            m_is_allocated_cpu = true;
        }
        if(allocate_gpu) {
            if(data_size == 0) {
                m_data_gpu = NULL;
            } else {
                ORcudaSafeCall(cudaMalloc(static_cast<void*>(&m_data_gpu), data_size * sizeof(T)));
            }
            m_is_allocated_gpu = true;
        }
    }

    template <typename T>
    void MemoryManager<T>::Free() {
        if(m_is_allocated_cpu) {
            if(m_data_cpu != NULL) {
                delete[] m_data_cpu;
            }
            m_is_allocated_cpu = false;
        }
        if(m_is_allocated_gpu) {
            if(m_data_gpu != NULL) {
                ORcudaSafeCall(cudaFree(m_data_cpu));
            }
            m_is_allocated_gpu = false;
        }
    }

    template <typename T>
    void MemoryManager<T>::InitData(unsigned char default_value = 0) {
        if(m_is_allocated_cpu) {
            memset(m_data_cpu, default_value, m_data_size * sizeof(T));
        }
        if(m_is_allocated_gpu) {
            ORcudaSafeCall(cudaMemset(m_data_gpu, default_value, m_data_size * sizeof(T)));
        }
    }

    template <typename T>
    void MemoryManager<T>::Swap(MemoryManager<T>& rhs) {
        std::swap(m_data_size, rhs.m_data_size);
        std::swap(m_data_cpu, rhs.m_data_cpu);
        std::swap(m_data_gpu, rhs.m_data_gpu);
        std::swap(m_is_allocated_cpu, rhs.m_is_allocated_cpu);
        std::swap(m_is_allocated_gpu, rhs.m_is_allocated_gpu);
    }

    template <typename T>
    void MemoryManager<T>::Resize(size_t data_size, bool reallocation) {
        if(data_size == m_data_size) {
            return;
        }
        if(data_size > m_data_size || reallocation) {
            bool allocate_cpu = m_is_allocated_cpu;
            bool allocate_gpu = m_is_allocated_gpu;

            Free();
            Allocate(data_size, allocate_cpu, allocate_gpu);
        }
    }


    template <typename T>
    void MemoryManager<T>::SetData(const MemoryManager<T> *source, MemoryCopyDirection copy_direction) {
        Resize(source->m_data_size);
        switch (copy_direction) {
            case CPU_TO_CPU:
                memcpy(this->m_data_cpu, source->m_data_cpu, m_data_size * sizeof(T));
                break;
            case CPU_TO_GPU:
                ORcudaSafeCall(cudaMemcpyAsync(m_data_gpu, source->m_data_cpu, source->dataSize * sizeof(T), cudaMemcpyHostToDevice));
                break;
            case GPU_TO_CPU:
                ORcudaSafeCall(cudaMemcpyAsync(m_data_cpu, source->m_data_gpu, source->dataSize * sizeof(T), cudaMemcpyDeviceToHost));
                break;
            case GPU_TO_GPU:
                ORcudaSafeCall(cudaMemcpyAsync(m_data_gpu, source->m_data_gpu, source->dataSize * sizeof(T), cudaMemcpyDeviceToDevice));
                break;
            default:
                break;
        }
    }

    template <typename T>
    T MemoryManager<T>::GetElement(int idx, MemoryDeviceType memory_type) const {
        switch (memory_type) {
            case MEMORYDEVICE_CPU:
                return m_data_cpu[idx];
            case MEMORYDEVICE_CUDA:
                T result;
                ORcudaSafeCall(cudaMemcpy(&result, this->data_cuda + n, sizeof(T), cudaMemcpyDeviceToHost));
                return result;
            default:
                throw std::runtime_error("Invalid memory type.");
        }
    }

    template <typename T>
    void MemoryManager<T>::CopyDataFromDeviceToHost() const {

    }

    template <typename T>
    void MemoryManager<T>::CopyDataFromHostToDevice() const {

    }
}



#endif //INFINITAM_MEMORYMANAGER_H
