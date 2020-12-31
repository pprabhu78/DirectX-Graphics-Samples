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

#include "stdafx.h"
#include "D3D12HelloTriangle.h"
#include "DXRHelper.h"
#include <iostream>

#include "RaytracingPipelineGenerator.h"
#include "RootSignatureGenerator.h"

D3D12HelloTriangle::D3D12HelloTriangle(UINT width, UINT height, std::wstring name) :
   DXSample(width, height, name),
   m_frameIndex(0),
   m_viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height)),
   m_scissorRect(0, 0, static_cast<LONG>(width), static_cast<LONG>(height)),
   m_rtvDescriptorSize(0),
   m_raster(false)

{
}

D3D12HelloTriangle::AccelerationStructureBuffer D3D12HelloTriangle::createBottomLevelAS(std::vector<std::pair<ComPtr<ID3D12Resource>, uint32_t>> vVertexBuffers)
{
   nv_helpers_dx12::BottomLevelASGenerator generator;
   // Adding all vertex buffers and not transforming their position.
   for (const auto &buffer : vVertexBuffers) {
      generator.AddVertexBuffer(buffer.first.Get(), 0, buffer.second,
         sizeof(Vertex), 0, 0);
   }

   std::uint64_t scratchSizeInBytes = 0;
   std::uint64_t resultSizeInBytes = 0;

   generator.ComputeASBufferSizes(myDevice.Get(), false, &scratchSizeInBytes, &resultSizeInBytes);

   AccelerationStructureBuffer buffers;

   buffers.pScratch = nv_helpers_dx12::CreateBuffer(
      myDevice.Get(), scratchSizeInBytes,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
      nv_helpers_dx12::kDefaultHeapProps);

   buffers.pResult = nv_helpers_dx12::CreateBuffer(
      myDevice.Get(), resultSizeInBytes,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
      nv_helpers_dx12::kDefaultHeapProps);

   generator.Generate(myCommandList.Get()
      , buffers.pScratch.Get(), buffers.pResult.Get()
      , false, nullptr);

   return buffers;
}

void D3D12HelloTriangle::CreateTopLevelAS(VectorInstances& instances)
{
   for (int i = 0; i < instances.size(); ++i)
   {
      const std::pair< ComPtr<ID3D12Resource>, DirectX::XMMATRIX >& instance = instances[i];
      myTopLevelGenerator.AddInstance(instance.first.Get(), instance.second, i, 0);
   }

   std::uint64_t scratchSizeInBytes = 0;
   std::uint64_t resultSizeInBytes = 0;
   std::uint64_t instanceDescsSize = 0;

   myTopLevelGenerator.ComputeASBufferSizes(myDevice.Get(), true, &scratchSizeInBytes, &resultSizeInBytes, &instanceDescsSize);

   myTLAS.pScratch = nv_helpers_dx12::CreateBuffer(
      myDevice.Get(), resultSizeInBytes,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
      nv_helpers_dx12::kDefaultHeapProps);

   myTLAS.pResult = nv_helpers_dx12::CreateBuffer(
      myDevice.Get(), resultSizeInBytes,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,

      nv_helpers_dx12::kDefaultHeapProps);

   myTLAS.pInstanceDesc = nv_helpers_dx12::CreateBuffer(
      myDevice.Get(), instanceDescsSize, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

   myTopLevelGenerator.Generate(myCommandList.Get()
      , myTLAS.pScratch.Get(), myTLAS.pResult.Get(), myTLAS.pInstanceDesc.Get(), nullptr);
}

void D3D12HelloTriangle::CreateAccelerationStructures()
{
   AccelerationStructureBuffer blas = createBottomLevelAS({ { myVertexBuffer.Get(), 3 } });
   myInstances = { {blas.pResult, XMMatrixIdentity()} };
   CreateTopLevelAS(myInstances);

   // Flush the command list and wait for it to finish
   myCommandList->Close();
   ID3D12CommandList *ppCommandLists[] = { myCommandList.Get() };
   m_commandQueue->ExecuteCommandLists(1, ppCommandLists);
   m_fenceValue++;
   m_commandQueue->Signal(m_fence.Get(), m_fenceValue);

   m_fence->SetEventOnCompletion(m_fenceValue, m_fenceEvent);
   WaitForSingleObject(m_fenceEvent, INFINITE);

   // Once the command list is finished executing, reset it to be reused for
   // rendering
   ThrowIfFailed(
      myCommandList->Reset(m_commandAllocator.Get(), m_pipelineState.Get()));

   // Store the AS buffers. The rest of the buffers will be released once we exit
   // the function
   myBlas = blas.pResult;
}

ComPtr<ID3D12RootSignature> D3D12HelloTriangle::CreateRayGenSignature(void)
{
   nv_helpers_dx12::RootSignatureGenerator generator;
   generator.AddHeapRangesParameter({
   { 0, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0}
 , { 0, 1, 0, D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1 }
   }
   );
   return generator.Generate(myDevice.Get(), true);
}

//-----------------------------------------------------------------------------
// The hit shader communicates only through the ray payload, and therefore does
// not require any resources
//
ComPtr<ID3D12RootSignature> D3D12HelloTriangle::CreateHitSignature(void)
{
   nv_helpers_dx12::RootSignatureGenerator generator;
   return generator.Generate(myDevice.Get(), true);
}

//-----------------------------------------------------------------------------
// The miss shader communicates only through the ray payload, and therefore
// does not require any resources
//
ComPtr<ID3D12RootSignature> D3D12HelloTriangle::CreateMissSignature(void)
{
   nv_helpers_dx12::RootSignatureGenerator generator;
   return generator.Generate(myDevice.Get(), true);
}

void D3D12HelloTriangle::CreateRaytracingPipeline(void)
{
   // The pipeline contains the DXIL code of all the shaders potentially executed
   // during the raytracing process. This section compiles the HLSL code into a
   // set of DXIL libraries. We chose to separate the code in several libraries
   // by semantic (ray generation, hit, miss) for clarity. Any code layout can be
   // used.
   myRayGenLibrary = nv_helpers_dx12::CompileShaderLibrary(L"RayGen.hlsl");
   myHitLibrary = nv_helpers_dx12::CompileShaderLibrary(L"Hit.hlsl");
   myMissLibrary = nv_helpers_dx12::CompileShaderLibrary(L"Miss.hlsl");

   // In a way similar to DLLs, each library is associated with a number of
   // exported symbols. This
   // has to be done explicitly in the lines below. Note that a single library
   // can contain an arbitrary number of symbols, whose semantic is given in HLSL
   // using the [shader("xxx")] syntax
   nv_helpers_dx12::RayTracingPipelineGenerator pipeLineGenerator(myDevice.Get());
   pipeLineGenerator.AddLibrary(myRayGenLibrary.Get(), { L"RayGen" });
   pipeLineGenerator.AddLibrary(myHitLibrary.Get(), { L"ClosestHit" });
   pipeLineGenerator.AddLibrary(myMissLibrary.Get(), { L"Miss" });

   // To be used, each DX12 shader needs a root signature defining which
   // parameters and buffers will be accessed.
   myRayGenSignature = CreateRayGenSignature();
   myHitSignature = CreateHitSignature();
   myMissSignature = CreateMissSignature();

   pipeLineGenerator.AddHitGroup(L"HitGroup", L"ClosestHit");

   pipeLineGenerator.AddRootSignatureAssociation(myRayGenSignature.Get(), { L"RayGen" });
   pipeLineGenerator.AddRootSignatureAssociation(myMissSignature.Get(), { L"Miss" });
   pipeLineGenerator.AddRootSignatureAssociation(myHitSignature.Get(), { L"HitGroup" });

   // The payload size defines the maximum size of the data carried by the rays,
   // ie. the the data
   // exchanged between shaders, such as the HitInfo structure in the HLSL code.
   // It is important to keep this value as low as possible as a too high value
   // would result in unnecessary memory consumption and cache trashing.
   pipeLineGenerator.SetMaxPayloadSize(4 * sizeof(float)); // RGB + distance

   // Upon hitting a surface, DXR can provide several attributes to the hit. In
   // our sample we just use the barycentric coordinates defined by the weights
   // u,v of the last two vertices of the triangle. The actual barycentrics can
   // be obtained using float3 barycentrics = float3(1.f-u-v, u, v);
   pipeLineGenerator.SetMaxAttributeSize(2 * sizeof(float)); // barycentric coordinates

   // The ray tracing process can shoot rays from existing hit points, resulting
   // in nested TraceRay calls. Our sample code traces only primary rays, which
   // then requires a trace depth of 1. Note that this recursion depth should be
   // kept to a minimum for best performance. Path tracing algorithms can be
   // easily flattened into a simple loop in the ray generation.
   pipeLineGenerator.SetMaxRecursionDepth(1);

   myRayTracingStateObject = pipeLineGenerator.Generate();

   ThrowIfFailed(myRayTracingStateObject->QueryInterface(IID_PPV_ARGS(&myRayTracingStateObjectProperties)));
}

void D3D12HelloTriangle::CreateRaytracingOutputBuffer()
{
   D3D12_RESOURCE_DESC resDesc = {};
   resDesc.DepthOrArraySize = 1;
   resDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
   // The back buffer is actually DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, but sRGB
   // formats cannot be used with UAVs. For accuracy we should convert to sRGB
   // ourselves in the shader
   resDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;

   resDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
   resDesc.Width = GetWidth();
   resDesc.Height = GetHeight();
   resDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
   resDesc.MipLevels = 1;
   resDesc.SampleDesc.Count = 1;
   ThrowIfFailed(myDevice->CreateCommittedResource(
      &nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &resDesc,
      D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
      IID_PPV_ARGS(&myRayTracingOutputBuffer)));
}

void D3D12HelloTriangle::CreateShaderResourceHeap(void)
{
   mySrvUavHeap = nv_helpers_dx12::CreateDescriptorHeap(
      myDevice.Get(), 2, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, true);

   // Get a handle to the heap memory on the CPU side, to be able to write the
   // descriptors directly
   D3D12_CPU_DESCRIPTOR_HANDLE srvHandle =
      mySrvUavHeap->GetCPUDescriptorHandleForHeapStart();

   // Create the UAV. Based on the root signature we created it is the first
   // entry. The Create*View methods write the view information directly into
   // srvHandle
   D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
   uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
   myDevice->CreateUnorderedAccessView(myRayTracingOutputBuffer.Get(), nullptr, &uavDesc,
      srvHandle);

   // Add the Top Level AS SRV right after the raytracing output buffer
   srvHandle.ptr += myDevice->GetDescriptorHandleIncrementSize(
      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
   D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc;
   srvDesc.Format = DXGI_FORMAT_UNKNOWN;
   srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
   srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
   srvDesc.RaytracingAccelerationStructure.Location =
      myTLAS.pResult->GetGPUVirtualAddress();
   // Write the acceleration structure view in the heap
   myDevice->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);
}

void D3D12HelloTriangle::CreateShaderBindingTable()
{
   myShaderBindingTableGenerator.Reset();

   D3D12_GPU_DESCRIPTOR_HANDLE srvUavHeapHandle =
      mySrvUavHeap->GetGPUDescriptorHandleForHeapStart();

   auto heapPointer = reinterpret_cast<std::uint64_t*>(srvUavHeapHandle.ptr);

   myShaderBindingTableGenerator.AddRayGenerationProgram(L"RayGen", { heapPointer });
   myShaderBindingTableGenerator.AddMissProgram(L"Miss", {});
   myShaderBindingTableGenerator.AddHitGroup(L"HitGroup", {});

   std::uint32_t sbtSize = myShaderBindingTableGenerator.ComputeSBTSize();
   myShaderBindingTableStorage = nv_helpers_dx12::CreateBuffer(
      myDevice.Get(), sbtSize, D3D12_RESOURCE_FLAG_NONE,
      D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
   if (!myShaderBindingTableStorage) {
      throw std::logic_error("Could not allocate the shader binding table");
   }

   myShaderBindingTableGenerator.Generate(myShaderBindingTableStorage.Get(), myRayTracingStateObjectProperties.Get());
}

void D3D12HelloTriangle::OnInit()
{
   LoadPipeline();
   LoadAssets();

   CheckRaytracingSupport();

   // Setup the acceleration structures (AS) for raytracing. When setting up
   // geometry, each bottom-level AS has its own transform matrix.
   CreateAccelerationStructures();

   CreateRaytracingPipeline();

   CreateRaytracingOutputBuffer();

   CreateShaderResourceHeap();

   CreateShaderBindingTable();

   // Command lists are created in the recording state, but there is nothing
   // to record yet. The main loop expects it to be closed, so close it now.
   ThrowIfFailed(myCommandList->Close());

}

bool D3D12HelloTriangle::CheckRaytracingSupport()
{
   D3D12_FEATURE_DATA_D3D12_OPTIONS5 options5 = {};
   myDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,
      &options5, sizeof(options5));
   if (options5.RaytracingTier < D3D12_RAYTRACING_TIER_1_0)
   {
      std::cout << "Ray tracing not supported!";
      return false;
   }
   return true;
}

void D3D12HelloTriangle::OnKeyUp(UINT8 key)
{
   m_raster = !m_raster;
}

// Load the rendering pipeline dependencies.
void D3D12HelloTriangle::LoadPipeline()
{
   UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
   // Enable the debug layer (requires the Graphics Tools "optional feature").
   // NOTE: Enabling the debug layer after device creation will invalidate the active device.
   {
      ComPtr<ID3D12Debug> debugController;
      if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
      {
         debugController->EnableDebugLayer();

         // Enable additional debug layers.
         dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
      }
   }
#endif

   ComPtr<IDXGIFactory4> factory;
   ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

   if (m_useWarpDevice)
   {
      ComPtr<IDXGIAdapter> warpAdapter;
      ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

      ThrowIfFailed(D3D12CreateDevice(
         warpAdapter.Get(),
         D3D_FEATURE_LEVEL_12_1,
         IID_PPV_ARGS(&myDevice)
      ));
   }
   else
   {
      ComPtr<IDXGIAdapter1> hardwareAdapter;
      GetHardwareAdapter(factory.Get(), &hardwareAdapter);

      ThrowIfFailed(D3D12CreateDevice(
         hardwareAdapter.Get(),
         D3D_FEATURE_LEVEL_11_0,
         IID_PPV_ARGS(&myDevice)
      ));
   }

   // Describe and create the command queue.
   D3D12_COMMAND_QUEUE_DESC queueDesc = {};
   queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
   queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

   ThrowIfFailed(myDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

   // Describe and create the swap chain.
   DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
   swapChainDesc.BufferCount = FrameCount;
   swapChainDesc.Width = m_width;
   swapChainDesc.Height = m_height;
   swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
   swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
   swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
   swapChainDesc.SampleDesc.Count = 1;

   ComPtr<IDXGISwapChain1> swapChain;
   ThrowIfFailed(factory->CreateSwapChainForHwnd(
      m_commandQueue.Get(),        // Swap chain needs the queue so that it can force a flush on it.
      Win32Application::GetHwnd(),
      &swapChainDesc,
      nullptr,
      nullptr,
      &swapChain
   ));

   // This sample does not support fullscreen transitions.
   ThrowIfFailed(factory->MakeWindowAssociation(Win32Application::GetHwnd(), DXGI_MWA_NO_ALT_ENTER));

   ThrowIfFailed(swapChain.As(&m_swapChain));
   m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

   // Create descriptor heaps.
   {
      // Describe and create a render target view (RTV) descriptor heap.
      D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
      rtvHeapDesc.NumDescriptors = FrameCount;
      rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
      rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
      ThrowIfFailed(myDevice->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));

      m_rtvDescriptorSize = myDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
   }

   // Create frame resources.
   {
      CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

      // Create a RTV for each frame.
      for (UINT n = 0; n < FrameCount; n++)
      {
         ThrowIfFailed(m_swapChain->GetBuffer(n, IID_PPV_ARGS(&m_renderTargets[n])));
         myDevice->CreateRenderTargetView(m_renderTargets[n].Get(), nullptr, rtvHandle);
         rtvHandle.Offset(1, m_rtvDescriptorSize);
      }
   }

   ThrowIfFailed(myDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_commandAllocator)));
}

// Load the sample assets.
void D3D12HelloTriangle::LoadAssets()
{
   // Create an empty root signature.
   {
      CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
      rootSignatureDesc.Init(0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

      ComPtr<ID3DBlob> signature;
      ComPtr<ID3DBlob> error;
      ThrowIfFailed(D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error));
      ThrowIfFailed(myDevice->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));
   }

   // Create the pipeline state, which includes compiling and loading shaders.
   {
      ComPtr<ID3DBlob> vertexShader;
      ComPtr<ID3DBlob> pixelShader;

#if defined(_DEBUG)
      // Enable better shader debugging with the graphics debugging tools.
      UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#else
      UINT compileFlags = 0;
#endif

      ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"shaders.hlsl").c_str(), nullptr, nullptr, "VSMain", "vs_5_0", compileFlags, 0, &vertexShader, nullptr));
      ThrowIfFailed(D3DCompileFromFile(GetAssetFullPath(L"shaders.hlsl").c_str(), nullptr, nullptr, "PSMain", "ps_5_0", compileFlags, 0, &pixelShader, nullptr));

      // Define the vertex input layout.
      D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
      {
      { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
      { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
      };

      // Describe and create the graphics pipeline state object (PSO).
      D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
      psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
      psoDesc.pRootSignature = m_rootSignature.Get();
      psoDesc.VS = CD3DX12_SHADER_BYTECODE(vertexShader.Get());
      psoDesc.PS = CD3DX12_SHADER_BYTECODE(pixelShader.Get());
      psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
      psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
      psoDesc.DepthStencilState.DepthEnable = FALSE;
      psoDesc.DepthStencilState.StencilEnable = FALSE;
      psoDesc.SampleMask = UINT_MAX;
      psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
      psoDesc.NumRenderTargets = 1;
      psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
      psoDesc.SampleDesc.Count = 1;
      ThrowIfFailed(myDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));
   }

   // Create the command list.
   ThrowIfFailed(myDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(), m_pipelineState.Get(), IID_PPV_ARGS(&myCommandList)));


   // Create the vertex buffer.
   {
      // Define the geometry for a triangle.
      Vertex triangleVertices[] =
      {
      { { 0.0f, 0.25f * m_aspectRatio, 0.0f }, { 1.0f, 0.0f, 0.0f, 1.0f } },
      { { 0.25f, -0.25f * m_aspectRatio, 0.0f }, { 0.0f, 1.0f, 0.0f, 1.0f } },
      { { -0.25f, -0.25f * m_aspectRatio, 0.0f }, { 0.0f, 0.0f, 1.0f, 1.0f } }
      };

      const UINT vertexBufferSize = sizeof(triangleVertices);

      // Note: using upload heaps to transfer static data like vert buffers is not 
      // recommended. Every time the GPU needs it, the upload heap will be marshalled 
      // over. Please read up on Default Heap usage. An upload heap is used here for 
      // code simplicity and because there are very few verts to actually transfer.
      ThrowIfFailed(myDevice->CreateCommittedResource(
         &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
         D3D12_HEAP_FLAG_NONE,
         &CD3DX12_RESOURCE_DESC::Buffer(vertexBufferSize),
         D3D12_RESOURCE_STATE_GENERIC_READ,
         nullptr,
         IID_PPV_ARGS(&myVertexBuffer)));

      // Copy the triangle data to the vertex buffer.
      UINT8* pVertexDataBegin;
      CD3DX12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
      ThrowIfFailed(myVertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
      memcpy(pVertexDataBegin, triangleVertices, sizeof(triangleVertices));
      myVertexBuffer->Unmap(0, nullptr);

      // Initialize the vertex buffer view.
      m_vertexBufferView.BufferLocation = myVertexBuffer->GetGPUVirtualAddress();
      m_vertexBufferView.StrideInBytes = sizeof(Vertex);
      m_vertexBufferView.SizeInBytes = vertexBufferSize;
   }

   // Create synchronization objects and wait until assets have been uploaded to the GPU.
   {
      ThrowIfFailed(myDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
      m_fenceValue = 1;

      // Create an event handle to use for frame synchronization.
      m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
      if (m_fenceEvent == nullptr)
      {
         ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
      }

      // Wait for the command list to execute; we are reusing the same command 
      // list in our main loop but for now, we just want to wait for setup to 
      // complete before continuing.
      WaitForPreviousFrame();
   }
}

// Update frame-based values.
void D3D12HelloTriangle::OnUpdate()
{
}

// Render the scene.
void D3D12HelloTriangle::OnRender()
{
   // Record all the commands we need to render the scene into the command list.
   PopulateCommandList();

   // Execute the command list.
   ID3D12CommandList* ppCommandLists[] = { myCommandList.Get() };
   m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

   // Present the frame.
   ThrowIfFailed(m_swapChain->Present(1, 0));

   WaitForPreviousFrame();
}

void D3D12HelloTriangle::OnDestroy()
{
   // Ensure that the GPU is no longer referencing resources that are about to be
   // cleaned up by the destructor.
   WaitForPreviousFrame();

   CloseHandle(m_fenceEvent);
}

void D3D12HelloTriangle::PopulateCommandList()
{
   // Command list allocators can only be reset when the associated 
   // command lists have finished execution on the GPU; apps should use 
   // fences to determine GPU execution progress.
   ThrowIfFailed(m_commandAllocator->Reset());

   // However, when ExecuteCommandList() is called on a particular command 
   // list, that command list can then be reset at any time and must be before 
   // re-recording.
   ThrowIfFailed(myCommandList->Reset(m_commandAllocator.Get(), m_pipelineState.Get()));

   // Set necessary state.
   myCommandList->SetGraphicsRootSignature(m_rootSignature.Get());
   myCommandList->RSSetViewports(1, &m_viewport);
   myCommandList->RSSetScissorRects(1, &m_scissorRect);

   // Indicate that the back buffer will be used as a render target.
   myCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

   CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
   myCommandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

   // Record commands.
   if (m_raster)
   {
      const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
      myCommandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
      myCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
      myCommandList->IASetVertexBuffers(0, 1, &m_vertexBufferView);
      myCommandList->DrawInstanced(3, 1, 0, 0);
   }
   else
   {
      std::vector<ID3D12DescriptorHeap*> heaps = { mySrvUavHeap.Get() };
      myCommandList->SetDescriptorHeaps((UINT)heaps.size(), heaps.data());

      // On the last frame, the ray tracing output was used as a copy source, to
      // copy its contents into the render target. Now we need to transition it to
      // a UAV so that the shaders can write in it.
      CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
         myRayTracingOutputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      myCommandList->ResourceBarrier(1, &transition);

      D3D12_DISPATCH_RAYS_DESC desc = {};

      // The layout of the SBT is as follows : ray generation shader, miss
      // shaders, hit groups. As described in the CreateShaderBindingTable method,
      // all SBT entries of a given type have the same size to allow a fixed stride.

      // The ray generation shaders are always at the beginning of the SBT. 
      const std::uint32_t rayGenSsectionSizeInBytes = myShaderBindingTableGenerator.GetRayGenSectionSize();
      desc.RayGenerationShaderRecord.StartAddress = myShaderBindingTableStorage->GetGPUVirtualAddress();
      desc.RayGenerationShaderRecord.SizeInBytes = rayGenSsectionSizeInBytes;


      const std::uint32_t missSectionSizeInBytes = myShaderBindingTableGenerator.GetMissSectionSize();
      desc.MissShaderTable.StartAddress = myShaderBindingTableStorage->GetGPUVirtualAddress() + rayGenSsectionSizeInBytes;
      desc.MissShaderTable.StrideInBytes = myShaderBindingTableGenerator.GetMissEntrySize();
      desc.MissShaderTable.SizeInBytes = missSectionSizeInBytes;

      desc.HitGroupTable.StartAddress = myShaderBindingTableStorage->GetGPUVirtualAddress() + rayGenSsectionSizeInBytes + missSectionSizeInBytes;
      desc.HitGroupTable.StrideInBytes = myShaderBindingTableGenerator.GetHitGroupEntrySize();
      const std::uint32_t hitGroupsSectionSize = myShaderBindingTableGenerator.GetHitGroupSectionSize();
      desc.HitGroupTable.SizeInBytes = myShaderBindingTableGenerator.GetHitGroupSectionSize();

      desc.Width = GetWidth();
      desc.Height = GetHeight();
      desc.Depth = 1;

      myCommandList->SetPipelineState1(myRayTracingStateObject.Get());
      myCommandList->DispatchRays(&desc);

      transition = CD3DX12_RESOURCE_BARRIER::Transition(
         myRayTracingOutputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
         D3D12_RESOURCE_STATE_COPY_SOURCE);
      myCommandList->ResourceBarrier(1, &transition);
      transition = CD3DX12_RESOURCE_BARRIER::Transition(
         m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
         D3D12_RESOURCE_STATE_COPY_DEST);
      myCommandList->ResourceBarrier(1, &transition);

      myCommandList->CopyResource(m_renderTargets[m_frameIndex].Get(),
         myRayTracingOutputBuffer.Get());

      transition = CD3DX12_RESOURCE_BARRIER::Transition(
         m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
         D3D12_RESOURCE_STATE_RENDER_TARGET);
      myCommandList->ResourceBarrier(1, &transition);

   }




   myCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_renderTargets[m_frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

   ThrowIfFailed(myCommandList->Close());
}

void D3D12HelloTriangle::WaitForPreviousFrame()
{
   // WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
   // This is code implemented as such for simplicity. The D3D12HelloFrameBuffering
   // sample illustrates how to use fences for efficient resource usage and to
   // maximize GPU utilization.

   // Signal and increment the fence value.
   const UINT64 fence = m_fenceValue;
   ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), fence));
   m_fenceValue++;

   // Wait until the previous frame is finished.
   if (m_fence->GetCompletedValue() < fence)
   {
      ThrowIfFailed(m_fence->SetEventOnCompletion(fence, m_fenceEvent));
      WaitForSingleObject(m_fenceEvent, INFINITE);
   }

   m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();
}
