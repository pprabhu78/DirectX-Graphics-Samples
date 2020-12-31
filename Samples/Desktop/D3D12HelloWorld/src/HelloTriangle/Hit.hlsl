#include "Common.hlsl"

struct Vertex
{
   float3 vertex;
   float4 color;
};

StructuredBuffer<Vertex> vertices : register(t0);


[shader("closesthit")] 
void ClosestHit(inout HitInfo payload, Attributes attrib) 
{
   float3 barycentrics =
      float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

   const float3 A = float3(1, 0, 0);
   const float3 B = float3(0, 1, 0);
   const float3 C = float3(0, 0, 1);


   uint vertId = 3 * PrimitiveIndex();

   float3 hitColor = vertices[vertId + 0].color.xyz * barycentrics.x
      + vertices[vertId + 1].color.xyz * barycentrics.y
      + vertices[vertId + 2].color.xyz * barycentrics.z;

   payload.colorAndDistance = float4(hitColor, RayTCurrent());
}
