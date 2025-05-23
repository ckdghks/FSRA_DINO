# Cross-view Geo-localization 기반 FSRA 실험 설계 구조 (업데이트됨)

이 문서는 FSRA 파이프라인에 DINO-style self-distillation 구조를 통합하여 성능 향상을 도모하고, 이후 patch-level 정렬과 모델 경량화를 점진적으로 실험하는 설계 구조입니다.

## 1. DINO-style Self-Distillation 구조 적용
- 기존 FSRA 파이프라인에 EMA 기반의 Teacher-Student 구조를 적용
- CLS-level embedding에 대해 distillation loss (e.g., cosine or MSE loss)를 적용
- 목적: 학습 안정성 향상 및 일반화 성능 확보

---

~~## 2. Patch-level Similarity Structure Loss 추가~~  
~~- Teacher와 Student 간 patch 간 similarity matrix를 정렬하도록 추가 loss를 구성~~  
~~- 목적: fine-grained 시맨틱 구조 보존 및 viewpoint-invariant 표현 강화~~  
❌ 실험 결과 성능이 나오지 않음

~~## 3. Backbone 경량화 (실시간성 확보)~~  
~~- ViT-B를 MobileViT 또는 EfficientFormer 등으로 교체~~  
~~- 기존 distill loss 및 patch-loss 유지~~  
~~- 목적: 정확도 손실 없이 추론 속도 및 메모리 사용량 개선~~  
❌ 실험 결과 성능이 나오지 않음

~~## 4. DINO Pretrained Backbone 실험 (선택적)~~  
~~- ViT를 DINO 방식으로 사전학습한 가중치로 초기화~~  
~~- 목적: 성능 부스팅 실험으로 사용하며, 필수 실험은 아님~~  
❌ 실험 결과 성능이 나오지 않음

---

이 실험 설계는 FSRA 기반 cross-view geo-localization 연구의 **최신 실험 구조**로 채택됩니다. 현재는 **1번 방식만이 유의미한 성능 향상**을 보였으며, 나머지 항목은 실험 결과에 따라 제외되었습니다.
