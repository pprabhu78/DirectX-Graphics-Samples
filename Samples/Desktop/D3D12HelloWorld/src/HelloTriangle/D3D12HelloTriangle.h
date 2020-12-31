//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#pragma once

#include "DXSample.h"

#include "TopLevelASGenerator.h"
#include "BottomLevelASGenerator.h"
#include "ShaderBindingTableGenerator.h"

#include <dxcapi.h>

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

class D3D12HelloTriangle : public DXSample
{
public:
    D3D12HelloTriangle(UINT width, UINT height, std::wstring name);

    virtual void OnInit();
    virtual void OnUpdate();
    virtual void OnRender();
    virtual void OnDestroy();

    virtual void OnKeyUp(UINT8 /*key*/) override;

private:
    static const UINT FrameCount = 2;

    struct Vertex
    {
        XMFLOAT3 position;
        XMFLOAT4 color;
    };

    // Pipeline objects.
    CD3DX12_VIEWPORT m_viewport;
    CD3DX12_RECT m_scissorRect;
    ComPtr<IDXGISwapChain3> m_swapChain;
    ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
    ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    ComPtr<ID3D12PipelineState> m_pipelineState;
    ComPtr<ID3D12Device5> myDevice;
    ComPtr<ID3D12GraphicsCommandList4> myCommandList;
    UINT m_rtvDescriptorSize;

    // App resources.
    ComPtr<ID3D12Resource> myVertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW myVertexBufferView;

    ComPtr<ID3D12Resource> myPlaneVertexBuffer;
    D3D12_VERTEX_BUFFER_VIEW myPlaneVertexBufferView;

    // Synchronization objects.
    UINT m_frameIndex;
    HANDLE m_fenceEvent;
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValue;

    bool m_raster;

    void LoadPipeline();
    void LoadAssets();
    void PopulateCommandList();
    void WaitForPreviousFrame();
    bool CheckRaytracingSupport();

    // The structure representing an AS
    struct AccelerationStructureBuffer
    {
       ComPtr<ID3D12Resource> pScratch;      // Scratch memory for AS builder
       ComPtr<ID3D12Resource> pResult;       // Where the AS is
       ComPtr<ID3D12Resource> pInstanceDesc; // Hold the matrices of the instances
    };

    // The instances
    typedef std::vector< std::pair< ComPtr<ID3D12Resource>, DirectX::XMMATRIX > > VectorInstances;
    VectorInstances myInstances;

    AccelerationStructureBuffer createBottomLevelAS(std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers);

    nv_helpers_dx12::TopLevelASGenerator myTopLevelGenerator;

    void CreateTopLevelAS(VectorInstances& instances);


    AccelerationStructureBuffer myTLAS;

    void CreateAccelerationStructures();

    ComPtr< ID3D12Resource> myBlas;


	ComPtr<ID3D12RootSignature> CreateRayGenSignature(void);
	ComPtr<ID3D12RootSignature> CreateHitSignature(void);
	ComPtr<ID3D12RootSignature> CreateMissSignature(void);

	ComPtr<ID3D12RootSignature> myRayGenSignature;
	ComPtr<ID3D12RootSignature> myHitSignature;
	ComPtr<ID3D12RootSignature> myMissSignature;
	ComPtr<IDxcBlob> myRayGenLibrary;
	ComPtr<IDxcBlob> myHitLibrary;
	ComPtr<IDxcBlob> myMissLibrary;

	// Ray tracing pipeline state
	ComPtr<ID3D12StateObject> myRayTracingStateObject;
	// Ray tracing pipeline state properties, retaining the shader identifiers
	// to use in the Shader Binding Table
	ComPtr<ID3D12StateObjectProperties> myRayTracingStateObjectProperties;

	void CreateRaytracingPipeline(void);

	void CreateRaytracingOutputBuffer();
	ComPtr<ID3D12Resource> myRayTracingOutputBuffer;

	void CreateShaderResourceHeap(void);
	ComPtr<ID3D12DescriptorHeap> mySrvUavHeap;

	void CreateShaderBindingTable();
	nv_helpers_dx12::ShaderBindingTableGenerator myShaderBindingTableGenerator;
	ComPtr<ID3D12Resource> myShaderBindingTableStorage;

   void createTrianglesVertexBuffer(void);
   void createPlaneVertexBuffer(void);

   // camera stufff
   void CreateCameraBuffer();
   void UpdateCameraBuffer();
   ComPtr<ID3D12Resource> myCameraBuffer;
   uint32_t myCameraBufferSize;

   ComPtr<ID3D12DescriptorHeap > myConstHeap;

   void OnButtonDown(UINT32 lParam) override;
   void OnMouseMove(UINT8 wParam, UINT32 lParam) override;
};
