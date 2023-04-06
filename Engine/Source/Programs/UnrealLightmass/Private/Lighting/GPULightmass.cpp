#include "CoreMinimal.h"
#include "Raster.h"
#include "LightingSystem.h"
#include "TextureMappingSetup.h"
#include "GPULightmassKernel.h"
#include "Async/ParallelFor.h"

namespace Lightmass
{

FCriticalSection GPULightmassCS;

void GPULightmassLogHandler(const wchar_t* message)
{
	UE_LOG(LogLightmass, Log, TEXT("%s"), message);
}

void FStaticLightingSystem::GenerateMapsForGPUSolver()
{
	FTaskGraphInterface::Startup(FPlatformMisc::NumberOfCores());

	GPULightmass::PreallocRadiositySurfaceCachePointers(AllMappings.Num());

	ParallelFor(AllMappings.Num(), [&](int32 MappingIndex)
	{
		FStaticLightingTextureMapping* TextureMapping = AllMappings[MappingIndex]->GetTextureMapping();

		if (TextureMapping)
		{
			GenerateAndExportMapsForGPUSolverTextureMapping(MappingIndex);
		}
	});

	size_t TotalTexels = 0;

	for (auto Mapping : AllMappings)
	{
		FStaticLightingTextureMapping* TextureMapping = Mapping->GetTextureMapping();

		if (TextureMapping && Mapping->GetVolumeMapping() == nullptr)
		{
			TotalTexels += TextureMapping->CachedSizeX * TextureMapping->CachedSizeY;
		}
	}

	GPULightmass::SetTotalTexelsForProgressReport(TotalTexels);
}

void FStaticLightingSystem::GenerateAndExportMapsForGPUSolverTextureMapping(int32 MappingIndex)
{
	FStaticLightingTextureMapping* TextureMapping = AllMappings[MappingIndex]->GetTextureMapping();

	FTexelToVertexMap TexelToVertexMap(TextureMapping->SurfaceCacheSizeX, TextureMapping->SurfaceCacheSizeY);

	bool bDebugThisMapping = false;
#if ALLOW_LIGHTMAP_SAMPLE_DEBUGGING
	bDebugThisMapping = TextureMapping == Scene.DebugMapping;
	int32 SurfaceCacheDebugX = -1;
	int32 SurfaceCacheDebugY = -1;
	if (bDebugThisMapping)
	{
		SurfaceCacheDebugX = FMath::TruncToInt(Scene.DebugInput.LocalX / (float)TextureMapping->CachedSizeX * TextureMapping->SurfaceCacheSizeX);
		SurfaceCacheDebugY = FMath::TruncToInt(Scene.DebugInput.LocalY / (float)TextureMapping->CachedSizeY * TextureMapping->SurfaceCacheSizeY);
	}
#endif

	RasterizeToSurfaceCacheTextureMapping(TextureMapping, bDebugThisMapping, TexelToVertexMap);

	TUniquePtr<FVector4f[]> WorldPositionMap{ new FVector4f[TextureMapping->SurfaceCacheSizeX * TextureMapping->SurfaceCacheSizeY]() };
	TUniquePtr<FVector4f[]> WorldNormalMap{ new FVector4f[TextureMapping->SurfaceCacheSizeX * TextureMapping->SurfaceCacheSizeY]() };
	TUniquePtr<FVector4f[]> ReflectanceMap{ new FVector4f[TextureMapping->SurfaceCacheSizeX * TextureMapping->SurfaceCacheSizeY]() };
	TUniquePtr<FVector4f[]> EmissiveMap{ new FVector4f[TextureMapping->SurfaceCacheSizeX * TextureMapping->SurfaceCacheSizeY]() };

	for (int32 Y = 0; Y < TextureMapping->SurfaceCacheSizeY; Y++)
	{
		for (int32 X = 0; X < TextureMapping->SurfaceCacheSizeX; X++)
		{
			bool bDebugThisTexel = false;
#if ALLOW_LIGHTMAP_SAMPLE_DEBUGGING
			if (bDebugThisMapping
				&& Y == SurfaceCacheDebugY
				&& X == SurfaceCacheDebugX)
			{
				bDebugThisTexel = true;
			}
#endif
			const FTexelToVertex& TexelToVertex = TexelToVertexMap(X, Y);
			if (TexelToVertex.TotalSampleWeight > 0.0f)
			{
				FFullStaticLightingVertex CurrentVertex = TexelToVertex.GetFullVertex();

				CurrentVertex.ApplyVertexModifications(TexelToVertex.ElementIndex, MaterialSettings.bUseNormalMapsForLighting, TextureMapping->Mesh);

				const int32 SurfaceCacheIndex = Y * TextureMapping->SurfaceCacheSizeX + X;

				WorldPositionMap[SurfaceCacheIndex] = CurrentVertex.WorldPosition;
				WorldNormalMap[SurfaceCacheIndex] = CurrentVertex.WorldTangentZ;

				const bool bIsTranslucent = TextureMapping->Mesh->IsTranslucent(TexelToVertex.ElementIndex);

				const FLinearColor Reflectance = (bIsTranslucent ? FLinearColor::Black : TextureMapping->Mesh->EvaluateTotalReflectance(CurrentVertex, TexelToVertex.ElementIndex));
				const FLinearColor Emissive = TextureMapping->Mesh->IsEmissive(TexelToVertex.ElementIndex) ? TextureMapping->Mesh->EvaluateEmissive(CurrentVertex.TextureCoordinates[0], TexelToVertex.ElementIndex) : FLinearColor::Black;

				ReflectanceMap[SurfaceCacheIndex] = FVector4f(FVector3f(Reflectance), TextureMapping->Mesh->IsTwoSided(TexelToVertex.ElementIndex));
				EmissiveMap[SurfaceCacheIndex] = FVector4f(FVector3f(Emissive), (!TextureMapping->Mesh->IsMasked(TexelToVertex.ElementIndex) || TextureMapping->Mesh->EvaluateMaskedCollision(CurrentVertex.TextureCoordinates[0], TexelToVertex.ElementIndex)) ? 1.0f : 0.0f);
			}
		}
	}

	{
		FScopeLock ScopeLock(&GPULightmassCS);

		GPULightmass::ImportSurfaceCache(
			MappingIndex,
			TextureMapping->SurfaceCacheSizeX,
			TextureMapping->SurfaceCacheSizeY,
			(GPULightmass::float4*)WorldPositionMap.Get(),
			(GPULightmass::float4*)WorldNormalMap.Get(),
			(GPULightmass::float4*)ReflectanceMap.Get(),
			(GPULightmass::float4*)EmissiveMap.Get()
		);
	}
}

void FStaticLightingSystem::ExportSkyCubemapToGPUSolver()
{
	const int32 NumThetaSteps = 128;
	const int32 NumPhiSteps = 128 * 2;

	TUniquePtr<FVector4f[]> UpperHemisphereCubemap{ new FVector4f[NumThetaSteps * NumPhiSteps]() };
	TUniquePtr<FVector4f[]> LowerHemisphereCubemap{ new FVector4f[NumThetaSteps * NumPhiSteps]() };

	if (Scene.SkyLights.Num() > 0)
	{
		const FSkyLight& SkyLight = Scene.SkyLights[0];

		FLMRandomStream RandomStream(123456);

		for (int32 ThetaIndex = 0; ThetaIndex < NumThetaSteps; ThetaIndex++)
		{
			for (int32 PhiIndex = 0; PhiIndex < NumPhiSteps; PhiIndex++)
			{
				const float U1 = RandomStream.GetFraction();
				const float U2 = RandomStream.GetFraction();

				const float Fraction1 = (ThetaIndex + U1) / (float)NumThetaSteps;
				const float Fraction2 = (PhiIndex + U2) / (float)NumPhiSteps;

				const float R = FMath::Sqrt(1.0f - Fraction1 * Fraction1);

				const float Phi = 2.0f * (float)PI * Fraction2;

				UpperHemisphereCubemap[ThetaIndex * NumPhiSteps + PhiIndex] = SkyLight.GetPathLighting(FVector4f(FMath::Cos(Phi) * R, FMath::Sin(Phi) * R, Fraction1), 0, true);
			}
		}

		for (int32 ThetaIndex = 0; ThetaIndex < NumThetaSteps; ThetaIndex++)
		{
			for (int32 PhiIndex = 0; PhiIndex < NumPhiSteps; PhiIndex++)
			{
				const float U1 = RandomStream.GetFraction();
				const float U2 = RandomStream.GetFraction();

				const float Fraction1 = (ThetaIndex + U1) / (float)NumThetaSteps;
				const float Fraction2 = (PhiIndex + U2) / (float)NumPhiSteps;

				const float R = FMath::Sqrt(1.0f - Fraction1 * Fraction1);

				const float Phi = 2.0f * (float)PI * Fraction2;

				LowerHemisphereCubemap[ThetaIndex * NumPhiSteps + PhiIndex] = SkyLight.GetPathLighting(FVector4f(FMath::Cos(Phi) * R, FMath::Sin(Phi) * R, -Fraction1), 0, true);
			}
		}
	}

	GPULightmass::ImportSkyLightCubemap(
		NumThetaSteps,
		NumPhiSteps,
		(GPULightmass::float4*)UpperHemisphereCubemap.Get(),
		(GPULightmass::float4*)LowerHemisphereCubemap.Get()
	);
}

void FStaticLightingSystem::ExportPunctualLightsToGPUSolver()
{
	TUniquePtr<GPULightmass::DirectionalLight[]> DirectionalLights{ new GPULightmass::DirectionalLight[Scene.DirectionalLights.Num()]() };

	for (int32 i = 0; i < Scene.DirectionalLights.Num(); i++)
	{
		DirectionalLights[i].Color.x = Scene.DirectionalLights[i].IndirectColor.R * Scene.DirectionalLights[i].Brightness;
		DirectionalLights[i].Color.y = Scene.DirectionalLights[i].IndirectColor.G * Scene.DirectionalLights[i].Brightness;
		DirectionalLights[i].Color.z = Scene.DirectionalLights[i].IndirectColor.B * Scene.DirectionalLights[i].Brightness;
		DirectionalLights[i].Direction.x = Scene.DirectionalLights[i].Direction.X;
		DirectionalLights[i].Direction.y = Scene.DirectionalLights[i].Direction.Y;
		DirectionalLights[i].Direction.z = Scene.DirectionalLights[i].Direction.Z;
	}

	TUniquePtr<GPULightmass::PointLight[]> PointLights { new GPULightmass::PointLight[Scene.PointLights.Num()]() };

	for (int32 i = 0; i < Scene.PointLights.Num(); i++)
	{
		PointLights[i].Color.x = Scene.PointLights[i].IndirectColor.R * Scene.PointLights[i].Brightness;
		PointLights[i].Color.y = Scene.PointLights[i].IndirectColor.G * Scene.PointLights[i].Brightness;
		PointLights[i].Color.z = Scene.PointLights[i].IndirectColor.B * Scene.PointLights[i].Brightness;
		PointLights[i].Radius = Scene.PointLights[i].Radius;
		PointLights[i].WorldPosition.x = Scene.PointLights[i].Position.X;
		PointLights[i].WorldPosition.y = Scene.PointLights[i].Position.Y;
		PointLights[i].WorldPosition.z = Scene.PointLights[i].Position.Z;
	}

	TUniquePtr<GPULightmass::SpotLight[]> SpotLights{ new GPULightmass::SpotLight[Scene.SpotLights.Num()]() };

	for (int32 i = 0; i < Scene.SpotLights.Num(); i++)
	{
		SpotLights[i].Color.x = Scene.SpotLights[i].IndirectColor.R * Scene.SpotLights[i].Brightness;
		SpotLights[i].Color.y = Scene.SpotLights[i].IndirectColor.G * Scene.SpotLights[i].Brightness;
		SpotLights[i].Color.z = Scene.SpotLights[i].IndirectColor.B * Scene.SpotLights[i].Brightness;
		SpotLights[i].Radius = Scene.SpotLights[i].Radius;
		SpotLights[i].WorldPosition.x = Scene.SpotLights[i].Position.X;
		SpotLights[i].WorldPosition.y = Scene.SpotLights[i].Position.Y;
		SpotLights[i].WorldPosition.z = Scene.SpotLights[i].Position.Z;
		SpotLights[i].CosInnerConeAngle = Scene.SpotLights[i].CosInnerConeAngle;
		SpotLights[i].CosOuterConeAngle = Scene.SpotLights[i].CosOuterConeAngle;
		SpotLights[i].Direction.x = Scene.SpotLights[i].Direction.X;
		SpotLights[i].Direction.y = Scene.SpotLights[i].Direction.Y;
		SpotLights[i].Direction.z = Scene.SpotLights[i].Direction.Z;
	}

	GPULightmass::ImportPunctualLights(Scene.DirectionalLights.Num(), DirectionalLights.Get(), Scene.PointLights.Num(), PointLights.Get(), Scene.SpotLights.Num(), SpotLights.Get());
}

#define CULL_NONSHADOWCASTING_TRIANGLES 1

TMap<FGuid, int32> MaterialGuidToIndexMap;
TArray<float*> MaskedCollisionMaps;

int32 CreateMaskedCollisionMap(FMaterial* Material)
{
	if (!MaterialGuidToIndexMap.Contains(Material->Guid))
	{
		int32 NewIndex = MaskedCollisionMaps.Num();
		MaskedCollisionMaps.Add(new float[Material->TransmissionSize * Material->TransmissionSize]());
		MaterialGuidToIndexMap.Add(Material->Guid, NewIndex);

		float* OutBuffer = MaskedCollisionMaps[NewIndex];

		for (int32 X = 0; X < Material->TransmissionSize; X++)
		{
			for (int32 Y = 0; Y < Material->TransmissionSize; Y++)
			{
				FVector2f UV((X + 0.5f) / Material->TransmissionSize, (Y + 0.5f) / Material->TransmissionSize);
				OutBuffer[Y * Material->TransmissionSize + X] = Material->SampleTransmission(UV).R > Material->OpacityMaskClipValue ? 1.0f : 0.0f;
			}
		}
	}

	return MaterialGuidToIndexMap[Material->Guid];
}

void FStaticLightingSystem::ExportAggregateMeshToGPUSolver()
{
	GPULightmass::SetLogHandler(GPULightmassLogHandler);

	GPULightmass::SetGlobalSamplingParameters(Scene.GPULightmassSettings.FireflyClampingThreshold);

	int32 NumVertices = 0;
	int32 NumTriangles = 0;

	for (int32 MappingIndex = 0; MappingIndex < AllMappings.Num(); MappingIndex++)
	{
		FStaticLightingTextureMapping* TextureMapping = AllMappings[MappingIndex]->GetTextureMapping();

		if (TextureMapping)
		{
			const FStaticLightingMesh* Mesh = TextureMapping->Mesh;

			if (Mesh->LightingFlags & GI_INSTANCE_CASTSHADOW)
			{
				NumVertices += Mesh->NumVertices;

#if CULL_NONSHADOWCASTING_TRIANGLES
				for (int32 TriangleIndex = 0; TriangleIndex < Mesh->NumTriangles; TriangleIndex++)
				{
					FStaticLightingVertex V0, V1, V2;
					int32 ElementIndex;
					Mesh->GetTriangle(TriangleIndex, V0, V1, V2, ElementIndex);

					if (Mesh->IsElementCastingShadow(ElementIndex) && !Mesh->IsTranslucent(ElementIndex))
					{
						NumTriangles++;
					}
				}
#else
				NumTriangles += Mesh->NumTriangles;
#endif
			}
		}
	}

	TUniquePtr<FVector3f[]> VertexWorldPositionBuffer{ new FVector3f[NumVertices]() };
	TUniquePtr<FVector2f[]> VertexTextureUVBuffer{ new FVector2f[NumVertices]() };
	TUniquePtr<FVector2f[]> VertexLightmapUVBuffer{ new FVector2f[NumVertices]() };
	TUniquePtr<int32[]> TriangleTextureMappingIndex{ new int32[NumTriangles]() };
	TUniquePtr<FIntVector[]> TriangleIndexBuffer{ new FIntVector[NumTriangles]() };
	TUniquePtr<int32[]> TriangleMaterialBuffer{ new int32[NumTriangles]() };

	int32 VertexOffset = 0;
	int32 TriangleOffset = 0;

	for (int32 MappingIndex = 0; MappingIndex < AllMappings.Num(); MappingIndex++)
	{
		FStaticLightingTextureMapping* TextureMapping = AllMappings[MappingIndex]->GetTextureMapping();

		if (TextureMapping)
		{
			const FStaticLightingMesh* Mesh = TextureMapping->Mesh;

			if (Mesh->LightingFlags & GI_INSTANCE_CASTSHADOW)
			{
				for (int32 TriangleIndex = 0; TriangleIndex < Mesh->NumTriangles; TriangleIndex++)
				{
					int32 I0 = 0, I1 = 0, I2 = 0;
					FStaticLightingVertex V0, V1, V2;
					int32 ElementIndex;
					Mesh->GetTriangleIndices(TriangleIndex, I0, I1, I2);
					Mesh->GetTriangle(TriangleIndex, V0, V1, V2, ElementIndex);

					VertexWorldPositionBuffer[VertexOffset + I0] = V0.WorldPosition;
					VertexWorldPositionBuffer[VertexOffset + I1] = V1.WorldPosition;
					VertexWorldPositionBuffer[VertexOffset + I2] = V2.WorldPosition;

					VertexTextureUVBuffer[VertexOffset + I0] = V0.TextureCoordinates[0];
					VertexTextureUVBuffer[VertexOffset + I1] = V1.TextureCoordinates[0];
					VertexTextureUVBuffer[VertexOffset + I2] = V2.TextureCoordinates[0];

					VertexLightmapUVBuffer[VertexOffset + I0] = V0.TextureCoordinates[TextureMapping->LightmapTextureCoordinateIndex];
					VertexLightmapUVBuffer[VertexOffset + I1] = V1.TextureCoordinates[TextureMapping->LightmapTextureCoordinateIndex];
					VertexLightmapUVBuffer[VertexOffset + I2] = V2.TextureCoordinates[TextureMapping->LightmapTextureCoordinateIndex];

#if CULL_NONSHADOWCASTING_TRIANGLES
					if (Mesh->IsElementCastingShadow(ElementIndex) && !Mesh->IsTranslucent(ElementIndex))
					{
						TriangleIndexBuffer[TriangleOffset] = FIntVector(VertexOffset + I0, VertexOffset + I1, VertexOffset + I2);
						TriangleTextureMappingIndex[TriangleOffset] = MappingIndex;
						if (TextureMapping->Mesh->IsMasked(ElementIndex))
						{
							TriangleMaterialBuffer[TriangleOffset] = CreateMaskedCollisionMap(TextureMapping->Mesh->GetMaterial(ElementIndex));
						}
						else
						{
							TriangleMaterialBuffer[TriangleOffset] = -1;
						}

						TriangleOffset++;
					}
#else
					TriangleIndexBuffer[TriangleOffset + TriangleIndex] = FIntVector(VertexOffset + I0, VertexOffset + I1, VertexOffset + I2);
					TriangleTextureMappingIndex[TriangleOffset + TriangleIndex] = MappingIndex;
#endif

				}

#if !CULL_NONSHADOWCASTING_TRIANGLES
				TriangleOffset += Mesh->NumTriangles;
#endif
				VertexOffset += Mesh->NumVertices;
			}
		}
	}

#if CULL_NONSHADOWCASTING_TRIANGLES
	check(NumTriangles == TriangleOffset);
#endif

	GPULightmass::ImportAggregateMesh(
		NumVertices,
		NumTriangles,
		(GPULightmass::float3*)VertexWorldPositionBuffer.Get(),
		(GPULightmass::float2*)VertexTextureUVBuffer.Get(),
		(GPULightmass::float2*)VertexLightmapUVBuffer.Get(),
		(GPULightmass::int3*)TriangleIndexBuffer.Get(),
		(int*)TriangleMaterialBuffer.Get(),
		(int*)TriangleTextureMappingIndex.Get()
	);

	GPULightmass::ImportMaterialMaps(
		MaskedCollisionMaps.Num(),
		Scene.MaterialSettings.TransmissionSize,
		MaskedCollisionMaps.GetData()
	);
}

void FStaticLightingSystem::SolveRadiosityOnGPU(int32 NumBounces)
{
	GPULightmass::RunRadiosity(NumBounces, Scene.GPULightmassSettings.NumSecondaryGISamples * 2);
}

void FStaticLightingSystem::CalculateIndirectLightingTextureMappingGPU(
	FStaticLightingTextureMapping* CurrentTextureMapping,
	bool bForceFlush)
{
	FScopeLock ScopeLock(&GPULightmassCS);

	if (CurrentTextureMapping)
	{
		TextureMappingsNeedToBeProcessedOnGPU.Push(CurrentTextureMapping);
		NumTexelsNeedToBeProcessedOnGPU += CurrentTextureMapping->CachedSizeX * CurrentTextureMapping->SizeY;
	}

	if (NumTexelsNeedToBeProcessedOnGPU >= 500 * 500 || bForceFlush)
	{
		static double lastFlushTime = FPlatformTime::Seconds();

		if (bForceFlush && (FPlatformTime::Seconds() - lastFlushTime < 2.0))
			return;

		if (TextureMappingsNeedToBeProcessedOnGPU.IsEmpty())
			return; // Nothing to flush

					// Dispatch

		const int32 BufferSizeSqrt = FMath::CeilToInt(FMath::Sqrt((float)NumTexelsNeedToBeProcessedOnGPU));
		const int32 BufferSize = BufferSizeSqrt * BufferSizeSqrt;
		TUniquePtr<FVector4f[]> BatchWorldPositionMap{ new FVector4f[BufferSize]() };
		TUniquePtr<FVector4f[]> BatchWorldNormalMap{ new FVector4f[BufferSize]() };
		TUniquePtr<float[]> BatchTexelRadiusMap{ new float[BufferSize]() };

		TArray<FStaticLightingTextureMapping*> Tasks;
		TextureMappingsNeedToBeProcessedOnGPU.PopAll(Tasks);

		int32 Offset = 0;

		size_t TotalTexels = 0;

		for (int32 i = 0; i < Tasks.Num(); i++)
		{
			FStaticLightingTextureMapping* TextureMapping = Tasks[i];

			TotalTexels += TextureMapping->CachedSizeX * TextureMapping->CachedSizeY;

			FTexelToVertexMap& TexelToVertexMap = *TextureMapping->TexelToVertexMap;

			TUniquePtr<FVector4f[]> WorldPositionMap{ new FVector4f[TextureMapping->CachedSizeX * TextureMapping->CachedSizeY]() };
			TUniquePtr<FVector4f[]> WorldNormalMap{ new FVector4f[TextureMapping->CachedSizeX * TextureMapping->CachedSizeY]() };
			TUniquePtr<float[]> TexelRadiusMap{ new float[TextureMapping->CachedSizeX * TextureMapping->CachedSizeY]() };

			{
				TUniquePtr<float[]> TexelRadiusMapBeforeFiltering{ new float[TextureMapping->CachedSizeX * TextureMapping->CachedSizeY]() };

				for (int32 Y = 0; Y < TextureMapping->CachedSizeY; Y++)
				{
					for (int32 X = 0; X < TextureMapping->CachedSizeX; X++)
					{
						bool bDebugThisTexel = false;
#if ALLOW_LIGHTMAP_SAMPLE_DEBUGGING
						if (bDebugThisMapping
							&& Y == SurfaceCacheDebugY
							&& X == SurfaceCacheDebugX)
						{
							bDebugThisTexel = true;
						}
#endif
						const FTexelToVertex& TexelToVertex = TexelToVertexMap(X, Y);
						if (TexelToVertex.TotalSampleWeight > 0.0f)
						{
							FFullStaticLightingVertex CurrentVertex = TexelToVertex.GetFullVertex();

							CurrentVertex.ApplyVertexModifications(TexelToVertex.ElementIndex, MaterialSettings.bUseNormalMapsForLighting, TextureMapping->Mesh);

							const int32 TexelIndex = Y * TextureMapping->CachedSizeX + X;

							WorldPositionMap[TexelIndex] = CurrentVertex.WorldPosition;
							WorldPositionMap[TexelIndex].W = TextureMapping->Mesh->IsTwoSided(TexelToVertex.ElementIndex) ? 1.0f : 0.0f;
							WorldNormalMap[TexelIndex] = CurrentVertex.WorldTangentZ;
							TexelRadiusMapBeforeFiltering[TexelIndex] = TexelToVertex.TexelRadius;

							//if (TextureMapping->Mesh->IsMasked(TexelToVertex.ElementIndex) && !TextureMapping->Mesh->EvaluateMaskedCollision(CurrentVertex.TextureCoordinates[0], TexelToVertex.ElementIndex))
							//	TexelRadiusMapBeforeFiltering[TexelIndex] = 0.0f;
						}
					}
				}

				for (int32 Y = 0; Y < TextureMapping->CachedSizeY; Y++)
				{
					for (int32 X = 0; X < TextureMapping->CachedSizeX; X++)
					{
						const int32 TexelIndex = Y * TextureMapping->CachedSizeX + X;
						if (TexelRadiusMapBeforeFiltering[TexelIndex] == 0.0f)
						{
							for (int32 dx = -1; dx <= 1; dx++)
							{
								for (int32 dy = -1; dy <= 1; dy++)
								{
									int32 Sx = X + dx;
									int32 Sy = Y + dy;
									if (Sx >= 0 && Sx < TextureMapping->CachedSizeX && Sy >= 0 && Sy < TextureMapping->CachedSizeY)
									{
										const int32 SourceTexelIndex = Sy * TextureMapping->CachedSizeX + Sx;
										if (TexelRadiusMapBeforeFiltering[SourceTexelIndex] != 0.0f)
										{
											TexelRadiusMap[TexelIndex] = TexelToVertexMap(X, Y).TexelRadius;
											break;
										}
									}
								}
							}
						}
						else {
							TexelRadiusMap[TexelIndex] = TexelRadiusMapBeforeFiltering[TexelIndex];
						}
					}
				}
			}

			FMemory::Memcpy(BatchWorldPositionMap.Get() + Offset, WorldPositionMap.Get(), TextureMapping->CachedSizeX * TextureMapping->CachedSizeY * sizeof(FVector4f));
			FMemory::Memcpy(BatchWorldNormalMap.Get() + Offset, WorldNormalMap.Get(), TextureMapping->CachedSizeX * TextureMapping->CachedSizeY * sizeof(FVector4f));
			FMemory::Memcpy(BatchTexelRadiusMap.Get() + Offset, TexelRadiusMap.Get(), TextureMapping->CachedSizeX * TextureMapping->CachedSizeY * sizeof(float));

			Offset += TextureMapping->CachedSizeX * TextureMapping->CachedSizeY;
		}

		TUniquePtr<GPULightmass::GatheredLightSample[]> OutLightmapData{ new GPULightmass::GatheredLightSample[BufferSize]() };

		GPULightmass::CalculateIndirectLightingTextureMapping(
			TotalTexels,
			BufferSizeSqrt,
			BufferSizeSqrt,
			Scene.GPULightmassSettings.NumPrimaryGISamples,
			(GPULightmass::float4*)BatchWorldPositionMap.Get(),
			(GPULightmass::float4*)BatchWorldNormalMap.Get(),
			BatchTexelRadiusMap.Get(),
			OutLightmapData.Get()
		);

		Offset = 0;

		for (int32 i = 0; i < Tasks.Num(); i++)
		{
			FStaticLightingTextureMapping* TextureMapping = Tasks[i];
			FGatheredLightMapData2D& LightMapData = *TextureMapping->LightMapData;

			for (uint32 Y = 0; Y < LightMapData.GetSizeY(); Y++)
			{
				for (uint32 X = 0; X < LightMapData.GetSizeX(); X++)
				{
					FGatheredLightMapSample& CurrentLightSample = LightMapData(X, Y);
					if (CurrentLightSample.bIsMapped)
					{
						FFinalGatherSample Sample;
						Sample.SetSkyOcclusion(*(FVector3f*)&OutLightmapData[Offset + Y * LightMapData.GetSizeX() + X].SkyOcclusion);
						Sample.SHVector = *(TSHVectorRGB<2>*)&OutLightmapData[Offset + Y * LightMapData.GetSizeX() + X].SHVector;
						Sample.SHCorrection = OutLightmapData[Offset + Y * LightMapData.GetSizeX() + X].SHCorrection;
						Sample.IncidentLighting = FLinearColor(*(FVector3f*)&OutLightmapData[Offset + Y * LightMapData.GetSizeX() + X].IncidentLighting);

						FFinalGatherSample AverageSample;
						float TotalNeighbourWeight = 0.0f;

						for (int32 dx = -1; dx <= 1; dx++)
							for (int32 dy = -1; dy <= 1; dy++)
							{
								if (dx == 0 && dy == 0) continue;
								int32 tx = X + dx;
								int32 ty = Y + dy;
								if (tx >= 0 && tx < (int32)LightMapData.GetSizeX() && ty >= 0 && ty < (int32)LightMapData.GetSizeY())
								{
									FFinalGatherSample TargetSample;
									TargetSample.SetSkyOcclusion(*(FVector3f*)&OutLightmapData[Offset + ty * LightMapData.GetSizeX() + tx].SkyOcclusion);
									TargetSample.SHVector = *(TSHVectorRGB<2>*)&OutLightmapData[Offset + ty * LightMapData.GetSizeX() + tx].SHVector;
									TargetSample.SHCorrection = OutLightmapData[Offset + ty * LightMapData.GetSizeX() + tx].SHCorrection;
									TargetSample.IncidentLighting = FLinearColor(*(FVector3f*)&OutLightmapData[Offset + ty * LightMapData.GetSizeX() + tx].IncidentLighting);
									AverageSample.AddWeighted(TargetSample, 1 - OutLightmapData[Offset + ty * LightMapData.GetSizeX() + tx].NumBackfaceHits);
									TotalNeighbourWeight += 1 - OutLightmapData[Offset + ty * LightMapData.GetSizeX() + tx].NumBackfaceHits;
								}
							}

						if (TotalNeighbourWeight > 0.0f)
							LightMapData(X, Y).AddWeighted(Sample * (1 - OutLightmapData[Offset + Y * LightMapData.GetSizeX() + X].NumBackfaceHits) + (AverageSample * (1 / TotalNeighbourWeight)) * OutLightmapData[Offset + Y * LightMapData.GetSizeX() + X].NumBackfaceHits, 1.0);
						else
							LightMapData(X, Y).AddWeighted(Sample, 1.0);
					}
				}
			}

			TextureMappingsNeedPostprocessing.Push(TextureMapping);

			Offset += TextureMapping->CachedSizeX * TextureMapping->CachedSizeY;
		}

		NumTexelsNeedToBeProcessedOnGPU = 0;

		lastFlushTime = FPlatformTime::Seconds();
	}
}

void FStaticLightingSystem::CalculateVolumetricLightmapBrickSamplesGPU(
	const int32 BrickSize,
	const FVector3f WorldBrickMin,
	const FVector3f WorldChildCellSize,
	TArray<FFinalGatherSample3>& OutUpperSamples,
	TArray<FFinalGatherSample3>& OutLowerSamples,
	TArray<float> OutMinDistances,
	TArray<float> OutBackfacingHitsFraction)
{
	FScopeLock ScopeLock(&GPULightmassCS);

	TArray<GPULightmass::VolumetricLightSample> UpperSamples;
	TArray<GPULightmass::VolumetricLightSample> LowerSamples;
	UpperSamples.AddZeroed(BrickSize*BrickSize*BrickSize);
	LowerSamples.AddZeroed(BrickSize*BrickSize*BrickSize);

	GPULightmass::CalculateVolumetricLightmapBrickSamples(
		BrickSize,
		*(GPULightmass::float3*)&WorldBrickMin,
		*(GPULightmass::float3*)&WorldChildCellSize,
		UpperSamples.GetData(),
		LowerSamples.GetData()
	);

	for (int32 i = 0; i < BrickSize*BrickSize*BrickSize; i++)
	{
		OutUpperSamples[i].SetSkyOcclusion(
			FVector3f(
				UpperSamples[i].SkyOcclusion.x,
				UpperSamples[i].SkyOcclusion.y,
				UpperSamples[i].SkyOcclusion.z
			)
		);

		for (int32 j = 0; j < 9; j++)
		{
			OutUpperSamples[i].SHVector.R.V[j] = UpperSamples[i].SHVector.r.v[j];
			OutUpperSamples[i].SHVector.G.V[j] = UpperSamples[i].SHVector.g.v[j];
			OutUpperSamples[i].SHVector.B.V[j] = UpperSamples[i].SHVector.b.v[j];
		}

		OutUpperSamples[i].IncidentLighting = FLinearColor(FVector3f(
			UpperSamples[i].IncidentLighting.x,
			UpperSamples[i].IncidentLighting.y,
			UpperSamples[i].IncidentLighting.z
		));

		OutLowerSamples[i].SetSkyOcclusion(
			FVector3f(
				LowerSamples[i].SkyOcclusion.x,
				LowerSamples[i].SkyOcclusion.y,
				LowerSamples[i].SkyOcclusion.z
			)
		);

		for (int32 j = 0; j < 9; j++)
		{
			OutLowerSamples[i].SHVector.R.V[j] = LowerSamples[i].SHVector.r.v[j];
			OutLowerSamples[i].SHVector.G.V[j] = LowerSamples[i].SHVector.g.v[j];
			OutLowerSamples[i].SHVector.B.V[j] = LowerSamples[i].SHVector.b.v[j];
		}

		OutLowerSamples[i].IncidentLighting = FLinearColor(FVector3f(
			LowerSamples[i].IncidentLighting.x,
			LowerSamples[i].IncidentLighting.y,
			LowerSamples[i].IncidentLighting.z
		));

		OutMinDistances[i] = FMath::Min(UpperSamples[i].MinDistance, LowerSamples[i].MinDistance);
		OutBackfacingHitsFraction[i] = .5f * (UpperSamples[i].BackfacingHitsFraction + LowerSamples[i].BackfacingHitsFraction);
	}
}

void FStaticLightingSystem::CalculateVolumeSampleListGPU(
	TArray<FVector3f> InWorldPositions,
	TArray<FFinalGatherSample3>& OutUpperSamples,
	TArray<FFinalGatherSample3>& OutLowerSamples)
{
	FScopeLock ScopeLock(&GPULightmassCS);

	TArray<GPULightmass::VolumetricLightSample> UpperSamples;
	TArray<GPULightmass::VolumetricLightSample> LowerSamples;
	UpperSamples.AddZeroed(InWorldPositions.Num());
	LowerSamples.AddZeroed(InWorldPositions.Num());

	GPULightmass::CalculateVolumeSampleList(InWorldPositions.Num(), (GPULightmass::float3*)InWorldPositions.GetData(), UpperSamples.GetData(), LowerSamples.GetData());

	for (int32 i = 0; i < InWorldPositions.Num(); i++)
	{
		OutUpperSamples[i].SetSkyOcclusion(
			FVector3f(
				UpperSamples[i].SkyOcclusion.x,
				UpperSamples[i].SkyOcclusion.y,
				UpperSamples[i].SkyOcclusion.z
			)
		);

		for (int32 j = 0; j < 9; j++)
		{
			OutUpperSamples[i].SHVector.R.V[j] = UpperSamples[i].SHVector.r.v[j];
			OutUpperSamples[i].SHVector.G.V[j] = UpperSamples[i].SHVector.g.v[j];
			OutUpperSamples[i].SHVector.B.V[j] = UpperSamples[i].SHVector.b.v[j];
		}

		OutUpperSamples[i].IncidentLighting = FLinearColor(FVector3f(
			UpperSamples[i].IncidentLighting.x,
			UpperSamples[i].IncidentLighting.y,
			UpperSamples[i].IncidentLighting.z
		));

		OutLowerSamples[i].SetSkyOcclusion(
			FVector3f(
				LowerSamples[i].SkyOcclusion.x,
				LowerSamples[i].SkyOcclusion.y,
				LowerSamples[i].SkyOcclusion.z
			)
		);

		for (int32 j = 0; j < 9; j++)
		{
			OutLowerSamples[i].SHVector.R.V[j] = LowerSamples[i].SHVector.r.v[j];
			OutLowerSamples[i].SHVector.G.V[j] = LowerSamples[i].SHVector.g.v[j];
			OutLowerSamples[i].SHVector.B.V[j] = LowerSamples[i].SHVector.b.v[j];
		}

		OutLowerSamples[i].IncidentLighting = FLinearColor(FVector3f(
			LowerSamples[i].IncidentLighting.x,
			LowerSamples[i].IncidentLighting.y,
			LowerSamples[i].IncidentLighting.z
		));
	}
}

}
