# autofoto

**사진 분류를 기기 안에서, 모델은 갈아 끼울 수 있게.** 서버에 사진을 올리지 않고 휴대폰에서
직접 추론합니다. 분류 모델은 앱에 박아두지 않고 실행 중에 목록을 받아 고르고 내려받습니다.

<!-- TODO: 스크린샷 2장을 docs/ 에 넣고 아래 주석을 해제하세요.
     ① 모델 목록에서 고르는 화면  ② 사진 선택 후 상위 3개 분류 결과 화면
     README에서 가장 먼저 읽히는 자리입니다. -->
<!-- | 모델 선택 | 분류 결과 |
|---|---|
| ![모델 선택](docs/model-picker.png) | ![분류 결과](docs/result.png) | -->

Flutter · TensorFlow Lite · MobileNetV2 (ImageNet 1000 클래스)

---

## 왜 온디바이스인가

사진 분류를 서버로 보내면 세 가지 비용이 생깁니다. 사용자 사진이 네트워크를 타고,
오프라인에서 못 쓰고, 사진 수만큼 추론 비용이 듭니다. 갤러리 정리처럼 **사진 수천 장을
한 번에 훑는 작업**에서는 이 세 가지가 다 커집니다.

그래서 추론을 기기로 내렸습니다. 다만 모델을 앱 번들에 넣으면 앱 용량이 커지고
모델을 바꿀 때마다 스토어 심사를 다시 받아야 합니다. **모델만 따로 내려받는 구조**로
그 둘을 분리했습니다.

---

## 구조

```
앱 시작
  │
  ├── GET /models                     서버에서 사용 가능한 모델 목록
  │
  ├── 사용자가 모델 선택
  │     └── GET /download-model?model_name=...   (zip)
  │           └── 압축 해제 → <앱문서>/models/<모델명>/{model.tflite, labels.txt}
  │                 └── 선택 상태를 shared_preferences에 저장
  │
  └── 사진 선택 (image_picker)
        └── 디코드 → 224×224 리사이즈 → [0,1] 정규화 → NHWC float32
              └── TFLite Interpreter.run()  ← 기기 내 추론
                    └── 상위 3개 클래스 + 확률
```

| 파일 | 역할 |
|---|---|
| `lib/services/image_classifier.dart` | TFLite 인터프리터 로드, 전처리, 추론, 상위 3개 추출 |
| `lib/api/api_service.dart` | 모델 목록 조회, zip 다운로드·해제, 모델별 폴더 관리 |
| `lib/screens/home_screen.dart` | 모델 선택 UI, 사진 선택, 결과 표시 |

모델은 앱 문서 디렉토리에 **모델명 폴더**로 나뉘어 저장되므로 여러 모델을 받아두고
바꿔 쓸 수 있습니다.

---

## 실행

```bash
flutter pub get
flutter run
```

모델 서버가 필요합니다. `GET /models`가 모델명 배열(JSON)을 돌려주고,
`GET /download-model?model_name=<name>`이 `model.tflite`와 `labels.txt`를 담은 zip을
돌려주면 됩니다.

```dart
// lib/api/api_service.dart
static const String _baseUrl = 'http://172.20.10.4:9000';
```

**주의: 이 주소가 하드코딩되어 있고 사설 IP입니다.** 개발 당시 휴대폰과 노트북을 같은
핫스팟에 붙여 테스트했기 때문입니다. 다른 환경에서 돌리려면 이 값을 바꿔야 합니다
(아래 「남은 일」의 첫 항목).

권한: iOS는 `ios/Runner/Info.plist`에 사진 접근 권한이, Android는 `permission_handler`로
런타임 요청이 들어가 있습니다.

---

## 측정

앱 우상단 **속도계 아이콘**을 누르면 선택한 사진으로 50여 회 추론해 구간별
**중앙값**을 표로 보여줍니다. 별도 프로파일러 없이 실기기에서 바로 읽을 수 있고,
나온 표를 아래에 그대로 붙이면 됩니다. 평균이 아니라 중앙값을 쓰는 이유는 모바일에서
GC·스케줄링 때문에 한두 번 크게 튀는 값이 섞이기 때문입니다.

일반 분류에서도 상태 줄에 구간별 시간이 한 줄로 찍힙니다
(`decode 12.4ms | resize 3.1ms | tensor 1.8ms | infer 24.6ms | ...`).

<!-- TODO: 실기기에서 속도계 버튼을 눌러 나온 표로 아래를 교체하세요.
     온디바이스 비전에서 가장 먼저 물어보는 숫자입니다. -->

| 항목 | 값 |
|---|---:|
| 테스트 기기 | TODO (예: iPhone 13, A15) |
| 모델 | MobileNetV2 1.0 224 (TFLite float32) |
| 디코드 | TODO ms |
| 리사이즈 | TODO ms |
| 텐서 변환 | TODO ms |
| **추론** | TODO ms |
| 전처리 합계 | TODO ms |
| 전체 | TODO ms |
| 모델 파일 크기 | 약 14MB |

전처리와 추론을 따로 재는 이유는, 온디바이스에서 **전처리가 추론보다 오래 걸리는
경우가 흔하기** 때문입니다. 추론 시간만 보고하면 절반만 보는 셈입니다.

### 전처리: 중첩 List → Float32List

처음 구현은 224×224×3 픽셀을 Dart 중첩 `List`에 한 픽셀씩 채웠습니다
(`List.generate` 4중첩 + 이중 루프 = 15만 회 대입). 이걸 `Float32List` 연속 쓰기로
바꿨는데, 여기에 한 단계가 더 있었습니다.

`Tensor.setTo()`에 `Float32List`를 그대로 넘기면 빨라지지 않습니다.
tflite_flutter의 `ByteConversionUtils.convertObjectToBytes`는 `Uint8List`와
`ByteBuffer`만 그대로 통과시키고, 그 밖의 `List`는 **원소마다 4바이트 버퍼를 새로
만들어** growable `List<int>`에 `addAll` 합니다. 즉 15만 번의 할당이 그대로 남습니다.
같은 메모리를 가리키는 `Uint8List` 뷰로 넘겨야 변환 없이 memcpy 한 번으로 끝납니다.

```dart
inputTensor.setTo(buffer.buffer.asUint8List());  // 뷰 — 복사 없음
```

두 경로를 다 남겨 뒀기 때문에 같은 기기에서 전/후를 비교할 수 있습니다
(`ImageClassifier.useFastPreprocess`, 속도계 버튼이 둘 다 측정).

| 텐서 변환 방식 | 소요 시간 |
|---|---:|
| 중첩 List (기존) | TODO ms |
| Float32List + Uint8List 뷰 | TODO ms |

### 정규화 범위 확인

MobileNetV2 배포본은 입력을 `[0,1]`로 기대하는 것과 `[-1,1]`로 기대하는 것이 섞여
있습니다. **틀려도 에러가 나지 않고 정확도만 조용히 떨어집니다.** 속도계 버튼이 같은
사진을 두 범위로 각각 분류해 상위 1개와 확률을 비교해 주므로, 맞는 쪽을 결과로
판단할 수 있습니다 (`InputNormalization`).

| 정규화 | 상위 1개 | 확률 |
|---|---|---:|
| zeroToOne | TODO | TODO |
| minusOneToOne | TODO | TODO |

---

## 남은 일 / 알려진 한계

- **서버 주소 하드코딩.** `--dart-define`이나 설정 화면으로 빼야 합니다. 지금은 같은
  핫스팟에 붙어야만 동작합니다
- **정규화 범위를 아직 확정하지 못했습니다.** 기본값은 `[0,1]`이고, 비교 도구는
  넣었지만 실기기 결과로 확정하는 일이 남았습니다 (위 「정규화 범위 확인」)
- **디코드가 남은 병목일 가능성이 큽니다.** 텐서 변환은 줄였지만 `img.decodeImage`는
  여전히 Dart 구현입니다. 측정 결과를 보고 `dart:ui` 디코더나 isolate로 옮길지
  판단할 계획입니다
- **다운로드 실패 처리가 얕습니다.** 네트워크 오류 시 빈 목록을 반환하고 로그만 남깁니다.
  재시도와 사용자 안내가 필요합니다
- **zip 무결성 검증 없음.** 체크섬 확인을 넣어야 합니다
- 양자화 모델(int8) 비교 미실시 — 크기·속도·정확도 트레이드오프를 재보고 싶습니다.
  지금 코드는 float32 입력만 빠른 경로를 쓰고, 그 외 타입은 예전 경로로 자동 폴백합니다
- **UI 스레드에서 추론합니다.** 사진 수천 장을 훑는 시나리오에서는 `IsolateInterpreter`로
  옮겨야 화면이 멈추지 않습니다

---

## 관련 작업

기기에서 돌아가는 비전 모델에 관심이 있어 이어서 만든 것들입니다.

- [**Bin_pind**](https://github.com/chaejoon23/Bin_pind) — 영상 프레임의 화질을 보고
  쓸 수 있는 프레임만 골라내는 파이프라인. 미니 ISP(화이트밸런스·톤·로컬 콘트라스트)로
  저조도 프레임을 복원해서 넘깁니다
- [**dinov3-image-search**](https://github.com/chaejoon23/DINOv3-image-similarity-search-system-TEST)
  — 자기지도 임베딩으로 닮은 이미지·장소를 찾는 검색

이 레포가 **배포**, Bin_pind가 **화질**, 위가 **표현학습** 쪽입니다.
