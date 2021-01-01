#include "Common.hlsl"

struct Vertex
{
   float3 vertex;
   float4 color;
};

StructuredBuffer<Vertex> vertices : register(t0);

cbuffer Colors : register(b0)
{
   float3 A;
}


[shader("closesthit")] 
void ClosestHit(inout HitInfo payload, Attributes attrib)
{
   float3 barycentrics =
      float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);


   uint vertId = 3 * PrimitiveIndex();

   float3 hitColor = float3(0.7, 0.7, 0.7);

   hitColor = A;

   payload.colorAndDistance = float4(hitColor, RayTCurrent());
}


[shader("closesthit")]
void PlaneClosestHit(inout HitInfo payload, Attributes attrib)
{
   float3 barycentrics =
      float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);


   uint vertId = 3 * PrimitiveIndex();

   float3 hitColor = float3(attrib.bary.x, attrib.bary.y, 1- attrib.bary.x- attrib.bary.y);

   payload.colorAndDistance = float4(hitColor, RayTCurrent());
}