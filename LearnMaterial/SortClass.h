#ifndef SORTCLASS_H
#define SORTCLASS_H
#include "Common.h"
//ģ���������ʵ��Ҫ����ͬһ���ļ�������
namespace SortSequence{
	class SortClass
	{
	public:
		SortClass();
		~SortClass();
		/******************************************************************************
		 �������ƣ� BuppleSort
		 ����˵���� ð�������ȶ����� ���n�o(n^2)
		 ��    ���� vector<T> 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void BuppleSort(vector<T>&nums);
		/******************************************************************************
		 �������ƣ� SelectSort
		 ����˵���� ֱ��ѡ���������ȶ����� o(n2)
		 ��    ���� vector<T> & nums 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void SelectSort(vector<T>&nums);
		/******************************************************************************
		 �������ƣ� SelectSort
		 ����˵���� �������� �ȶ����� n^2
		 ��    ���� vector<T> & nums 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void InsertSort(vector<T>&nums);
		/******************************************************************************
		 �������ƣ� ShellSort
		 ����˵���� ϣ������ ���ȶ�n^1.3
		 ��    ���� vector<T> & nums 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void ShellSort(vector<T>&nums);
		/******************************************************************************
		 �������ƣ� MergeSort
		 ����˵���� �鲢����-�ȶ����� nlogn 
		 ��    ���� vector<T> & nums 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void MergeSort(vector<T>&nums);
		/******************************************************************************
		 �������ƣ� QuickSort
		 ����˵���� ��������-���ȶ����� ƽ��nlogn
		 ��    ���� vector<T> & nums 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void QuickSort(vector<T>&nums);
		//�������ȶ�����
		/******************************************************************************
		 �������ƣ� HeapSort
		 ����˵���� �������ȶ����� nlogn
		 ��    ���� vector<T> & nums 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void HeapSort(vector<T>&nums);
		/******************************************************************************
		 �������ƣ� RadixSort
		 ����˵���� ���������ȶ����� n*logr*m
		 ��    ���� vector<T> & nums 
		 �� �� ֵ�� void
		 ��    �ߣ� Ru Long
		 ��    �ڣ� 2020/02/21
		******************************************************************************/
		template<typename T>
		void RadixSort(vector<T>& nums);

		SortClass & operator =(const SortClass& ist);
	private:
		template<typename T>
		void Merge(vector<T>& nums, vector<T>& helpNums, const int start, const int end);
		template<typename T>
		void PartionSort(vector<T>&nums, const int start,const int end);
		template<typename T>
		void HelpHeapSort(vector<T>&nums, size_t start, const size_t len);
		char * m_pData;
	};

	template<typename T>
	void SortSequence::SortClass::BuppleSort(vector<T>&nums)
	{
		if (nums.size() < 2)
			return;
		int length = nums.size() - 1;
		while (true)
		{
			bool bSwap = false;
			for (int i = 0; i < length; ++i)
			{
				if (nums[i]>nums[i + 1])
				{
					swap(nums[i], nums[i + 1]);
					bSwap = true;
				}
			}
			--length;
			if (!bSwap)
			{
				break;
			}
		}
	}

	template<typename T>
	void SortSequence::SortClass::SelectSort(vector<T>&nums)
	{
		if (nums.size() < 2)
		{
			return;
		}
		int end = nums.size();
		while (end)
		{
			int index = 0;
			for (int i = 0; i < end; ++i)
			{
				if (nums[i] > nums[index])
					index = i;
			}
			swap(nums[--end], nums[index]);
		}
	}

	template<typename T>
	void SortSequence::SortClass::InsertSort(vector<T>&nums)
	{
		if (nums.size() < 2)
			return;
		const int length = nums.size();
		for (int i = 1; i < length; ++i)
		{
			int indexTemp = i;
			while (indexTemp>0 && nums[indexTemp] < nums[indexTemp - 1])
			{
				swap(nums[indexTemp], nums[indexTemp - 1]);
				--indexTemp;
			}
		}
	}

	template<typename T>
	void SortClass::ShellSort(vector<T>&nums)
	{
		if (nums.size() < 2)
			return;
		const int length = nums.size();
		int gap = length;
		while (gap>1)
		{
			gap = gap / 3 + 1;
			for (int i = gap; i < length; i++)
			{
				int indexTemp = i;
				while (indexTemp >= gap&&nums[indexTemp] < nums[indexTemp - gap])
				{
					swap(nums[indexTemp], nums[indexTemp - gap]);
					indexTemp -= gap;
				}
			}
		}
	}

	template<typename T>
	void SortClass::MergeSort(vector<T>&nums)
	{
		if (nums.size() < 2)
			return;
		vector<T> helpNums(nums.size());
		Merge(nums, helpNums, 0, nums.size() - 1);
	}

	template<typename T>
	void SortClass::QuickSort(vector<T>&nums)
	{
		if (nums.size() < 2)
			return;
		PartionSort(nums, 0, nums.size()-1);
	}

	template<typename T>
	void SortClass::HelpHeapSort(vector<T>&nums, size_t start, const size_t len)
	{
		size_t nextChildIndex = 2 * start + 1;
		T tempNode = nums[start];
		while (nextChildIndex < len)
		{
			if (nextChildIndex + 1 < len&&nums[nextChildIndex] < nums[nextChildIndex + 1])
			{
				++nextChildIndex;
			}
			if (tempNode>nums[nextChildIndex])
			{
				break;
			}
			else
			{
				nums[start] = nums[nextChildIndex];
				start = nextChildIndex;
				nextChildIndex = 2 * start + 1;
			}
		}
		nums[start] = tempNode;
	}

	template<typename T>
	void SortClass::HeapSort(vector<T>&nums)
	{
		if (nums.size() < 2)
			return;
		const int length = nums.size();
		for (int i = length / 2 - 1; i >= 0; --i)
		{
			HelpHeapSort(nums, i, length);
		}
		for (int i = length - 1; i >= 0; --i)
		{
			swap(nums[i], nums[0]);
			HelpHeapSort(nums, 0, i);
		}
	}

	template<typename T>
	void SortClass::RadixSort(vector<T>& nums)
	{
		if (nums.size() < 2)
		{
			return;
		}
		const int length = nums.size();
		int max_Bit = 0;
		for (auto it : nums)
		{
			int countBit = 0;
			while (it)
			{
				++countBit;
				it /= 10;
			}
			max_Bit = max(max_Bit, countBit);
		}
		vector<queue<T>> vecQue(10);
		int mod = 10;
		int div = 1;
		for (int i = 0; i < max_Bit; ++i)
		{
			for (auto it : nums)
			{
				int iTemp = (it %mod) / div;
				vecQue[iTemp].push(it);
			}
			int index = 0;
			for (int j = 0; j < 10; ++j)
			{
				while (!vecQue[j].empty())
				{
					nums[index++] = vecQue[j].front();
					vecQue[j].pop();
				}
			}
			mod *= 10;
			div *= 10;
		}
	}

	template<typename T>
	void SortClass::Merge(vector<T>& nums, vector<T>& helpNums, const int start, const int end)
	{
		if (start == end)
		{
			helpNums[start] = nums[end];
			return;
		}
		const int length = (end - start) >> 1;
		Merge(nums, helpNums, start, start + length);
		Merge(nums, helpNums, start + length + 1, end);
		//�ϲ���������
		int index = end;
		int leftCur = start + length, rightCur = end;
		while (leftCur >= start&&rightCur >= start + length + 1)
		{
			if (nums[leftCur] < nums[rightCur])
			{
				helpNums[index--] = nums[rightCur--];
			}
			else
			{
				helpNums[index--] = nums[leftCur--];
			}
		}
		while (leftCur >= start)
		{
			helpNums[index--] = nums[leftCur--];
		}
		while (rightCur >= start + length + 1)
		{
			helpNums[index--] = nums[rightCur--];
		}
		for (int i = start; i <= end; ++i)
		{
			nums[i] = helpNums[i];
		}
	}

	template<typename T>
	void SortClass::PartionSort(vector<T>&nums, const int start, const int end)
	{
		//ֻ��һ����
		if (start >= end)
		{
			return;
		}
		int leftCur = start, rightCur = end+1;
		int proMid = nums[start];
		while (leftCur < rightCur)
		{
			do
			{
				++leftCur;
			} while (leftCur < rightCur&&nums[leftCur] < proMid);
			do
			{
				--rightCur;
			} while (nums[rightCur] > proMid);
			if (leftCur >= rightCur)
			{
				break;
			}
			swap(nums[leftCur], nums[rightCur]);
		}
		swap(nums[start], nums[rightCur]);
		PartionSort(nums, start, rightCur-1);
		PartionSort(nums, rightCur + 1, end);
	}
}
#endif 

