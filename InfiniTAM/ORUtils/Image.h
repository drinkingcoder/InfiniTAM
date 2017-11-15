// Copyright 2014-2017 Oxford University Innovation Limited and the authors of InfiniTAM

#pragma once

#include "MemoryManager.h"
#include "Vector.h"

#ifndef __METALC__

namespace ORUtils
{
	/** \brief
	Represents images, templated on the pixel type
	*/
	template <typename T>
	class Image : private MemoryManager<T>
	{
	public:
		/** Expose public MemoryBlock<T> member variables. */
		using MemoryManager<T>::m_data_size;

		/** Expose public MemoryBlock<T> datatypes. */
		using typename MemoryManager<T>::MemoryCopyDirection;
		using MemoryManager<T>::CPU_TO_CPU;
		using MemoryManager<T>::CPU_TO_CUDA;
		using MemoryManager<T>::CUDA_TO_CPU;
		using MemoryManager<T>::CUDA_TO_CUDA;

		/** Expose public MemoryBlock<T> member functions. */
		using MemoryManager<T>::InitData;
		using MemoryManager<T>::GetData;
		using MemoryManager<T>::GetElement;
#ifdef COMPILE_WITH_METAL
		using MemoryManager<T>::GetMetalBuffer();
#endif
		using MemoryManager<T>::UpdateDeviceFromHost;
		using MemoryManager<T>::UpdateHostFromDevice;

		/** Size of the image in pixels. */
		Vector2<int> noDims;

		/** Initialize an empty image of the given size, either
		on CPU only or on both CPU and GPU.
		*/
		Image(Vector2<int> noDims, bool allocate_CPU, bool allocate_CUDA)
			: MemoryManager<T>(noDims.x * noDims.y, allocate_CPU, allocate_CUDA)
		{
			this->noDims = noDims;
		}

		Image(bool allocate_CPU, bool allocate_CUDA)
			: MemoryManager<T>(0, allocate_CPU, allocate_CUDA)
		{
			this->noDims = Vector2<int>(0, 0);
		}

		Image(Vector2<int> noDims, MemoryDeviceType memoryType)
			: MemoryManager<T>(noDims.x * noDims.y, memoryType)
		{
			this->noDims = noDims;
		}

		/** Resize an image, losing all old image data.
		Essentially any previously allocated data is
		released, new memory is allocated.
		*/
		void ChangeDims(Vector2<int> newDims, bool forceReallocation = true)
		{
			MemoryManager<T>::Resize(newDims.x * newDims.y, forceReallocation);
			noDims = newDims;
		}

		void SetFrom(const Image<T> *source, MemoryCopyDirection memoryCopyDirection)
		{
			ChangeDims(source->noDims);
			MemoryManager<T>::SetData(source, memoryCopyDirection);
		}

		void Swap(Image<T>& rhs)
		{
			MemoryManager<T>::Swap(rhs);
			std::swap(this->noDims, rhs.noDims);
		}

		// Suppress the default copy constructor and assignment operator
		Image(const Image&) = delete;
		Image& operator=(const Image&) = delete;
	};
}

#endif
