// InfiniTAM: Surffuse. Copyright (c) Torr Vision Group and the authors of InfiniTAM, 2016.

#include "ITMSurfelSceneReconstructionEngine.h"

namespace ITMLib
{

//#################### CONSTRUCTORS ####################

template <typename TSurfel>
ITMSurfelSceneReconstructionEngine<TSurfel>::ITMSurfelSceneReconstructionEngine(const Vector2i& depthImageSize)
: m_timestamp(0)
{
  size_t pixelCount = depthImageSize.x * depthImageSize.y;
  m_correspondenceMapMB = new ORUtils::MemoryManager<unsigned int>(pixelCount, true, true);
  m_mergeTargetMapMB = new ORUtils::MemoryManager<unsigned int>(pixelCount, true, true);
  m_newPointsMaskMB = new ORUtils::MemoryManager<unsigned short>(pixelCount + 1, true, true);
  m_newPointsPrefixSumMB = new ORUtils::MemoryManager<unsigned int>(pixelCount + 1, true, true);
  m_normalMapMB = new ORUtils::MemoryManager<Vector3f>(pixelCount, true, true);
  m_radiusMapMB = new ORUtils::MemoryManager<float>(pixelCount, true, true);
  m_surfelRemovalMaskMB = new ORUtils::MemoryManager<unsigned int>(MAX_SURFEL_COUNT, true, true);
  m_vertexMapMB =  new ORUtils::MemoryManager<Vector4f>(pixelCount, true, true);

  // Make sure that the dummy element at the end of the new points mask is initialised properly.
  m_newPointsMaskMB->GetData(MEMORYDEVICE_CPU)[pixelCount] = 0;
  m_newPointsMaskMB->UpdateDeviceFromHost();
}

//#################### DESTRUCTOR ####################

template <typename TSurfel>
ITMSurfelSceneReconstructionEngine<TSurfel>::~ITMSurfelSceneReconstructionEngine()
{
  delete m_correspondenceMapMB;
  delete m_mergeTargetMapMB;
  delete m_newPointsMaskMB;
  delete m_newPointsPrefixSumMB;
  delete m_normalMapMB;
  delete m_radiusMapMB;
  delete m_surfelRemovalMaskMB;
  delete m_vertexMapMB;
}

//#################### PUBLIC MEMBER FUNCTIONS ####################

template <typename TSurfel>
void ITMSurfelSceneReconstructionEngine<TSurfel>::IntegrateIntoScene(ITMSurfelScene<TSurfel> *scene, const ITMView *view, const ITMTrackingState *trackingState,
                                                                     const ITMSurfelRenderState *renderState)
{
  PreprocessDepthMap(view, scene->GetParams());
  FindCorrespondingSurfels(scene, view, trackingState, renderState);
  FuseMatchedPoints(scene, view, trackingState);
  AddNewSurfels(scene, view, trackingState);
  MarkBadSurfels(scene);
  if(scene->GetParams().useSurfelMerging) MergeSimilarSurfels(scene, renderState);
  RemoveMarkedSurfels(scene);

  ++m_timestamp;
}

template <typename TSurfel>
void ITMSurfelSceneReconstructionEngine<TSurfel>::ResetScene(ITMSurfelScene<TSurfel> *scene) const
{
  scene->Reset();
}

}
