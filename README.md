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
        └── 디코드 → 224×224 리사이즈 → [-1,1] 정규화 → NHWC float32
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

모델 서버와 앱 두 개를 띄웁니다. 폰과 맥이 **같은 와이파이**에 있으면 됩니다.

```bash
# 1) 모델 받기 + 서버 (맥, 표준 라이브러리만 사용)
python3 tools/fetch_model.py        # tools/models/mobilenet_v2/{model.tflite,labels.txt}, SHA-256 확인
python3 tools/model_server.py       # 0.0.0.0:9000 — 앱 빌드 명령을 주소까지 채워 출력합니다

# 2) 앱 (폰 연결)
flutter pub get
flutter run --profile --dart-define=MODEL_SERVER=http://<맥 IP>:9000
```

서버 규약은 두 개뿐입니다. `GET /models`가 모델명 배열(JSON)을,
`GET /download-model?model_name=<name>`이 `model.tflite`와 `labels.txt`를 담은 zip을
돌려줍니다. 처음 만들 때는 서버 주소가 핫스팟 사설 IP로 하드코딩되어 있었고 그 서버와
모델 파일은 남아 있지 않아서, 주소를 `--dart-define`으로 빼고 서버를 레포에 넣었습니다.

**모델.** TensorFlow 공식 [tflite-support](https://github.com/tensorflow/tflite-support)
테스트 데이터의 `mobilenet_v2_1.0_224.tflite`로 고정했습니다 (float32, 1001 클래스,
14.0MB, SHA-256 `ff5cb7f9…`). 첫 커밋의 `labels.txt`가 이 모델의 내장 라벨과 1001줄
모두 같아서, 원래 쓰던 모델도 같은 계열이었을 가능성이 높습니다.

권한: iOS는 `ios/Runner/Info.plist`에 사진 접근·로컬 네트워크 권한이, Android는
`permission_handler`로 런타임 요청과 매니페스트의 `INTERNET`·평문 HTTP 허용이 들어가
있습니다.

---

## 측정

앱 우상단 **속도계 아이콘**을 누르면 선택한 사진으로 50여 회 추론해 구간별
**중앙값**을 표로 보여줍니다. **`--profile`이나 `--release` 빌드에서 재야 합니다.**
debug 빌드는 Dart가 JIT로 돌아 전처리 시간이 몇 배 부풀려지므로, 표 맨 위에 빌드 모드를
찍고 debug면 경고를 띄웁니다. 별도 프로파일러 없이 실기기에서 바로 읽을 수 있고,
나온 표를 아래에 그대로 붙이면 됩니다. 평균이 아니라 중앙값을 쓰는 이유는 모바일에서
GC·스케줄링 때문에 한두 번 크게 튀는 값이 섞이기 때문입니다.

일반 분류에서도 상태 줄에 구간별 시간이 한 줄로 찍힙니다
(`decode 12.4ms | resize 3.1ms | tensor 1.8ms | infer 24.6ms | ...`).

<!-- TODO: 실기기에서 속도계 버튼을 눌러 나온 표로 아래를 교체하세요.
     온디바이스 비전에서 가장 먼저 물어보는 숫자입니다. -->

| 항목 | 값 |
|---|---:|
| 테스트 기기 | TODO (예: iPhone 13, A15) |
| 빌드 | TODO (profile / release) |
| 모델 | MobileNetV2 1.0 224 (TFLite float32, 1001 클래스) |
| 정규화 | minusOneToOne |
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

### 정규화 범위 확인 — `[-1,1]`로 확정

MobileNetV2 배포본은 입력을 `[0,1]`로 기대하는 것과 `[-1,1]`로 기대하는 것이 섞여
있습니다. **틀려도 에러가 나지 않고 정확도만 조용히 떨어집니다.** 이 모델은 `[-1,1]`입니다.

1. **메타데이터.** 모델에 박힌 `NormalizationOptions`가 mean 127.5 / std 127.5
2. **정확도.** ImageNet 클래스당 1장, 1000장을 앱과 같은 전처리(크롭 없이 전체 리사이즈)로
   분류했습니다 ([`tools/check_normalization.py`](tools/check_normalization.py),
   결과 [`tools/results/normalization.json`](tools/results/normalization.json))

| 리사이즈 | 정규화 | top-1 | top-5 |
|---|---|---:|---:|
| nearest (앱 기본) | `[0,1]` | 75.0% | 92.8% |
| nearest (앱 기본) | **`[-1,1]`** | **83.9%** | **96.8%** |
| bilinear | `[0,1]` | 75.4% | 92.4% |
| bilinear | **`[-1,1]`** | **85.8%** | **97.8%** |

같은 사진 1000장을 짝지어 비교하면 `[-1,1]`만 맞힌 사진이 109장, `[0,1]`만 맞힌 사진이
20장입니다 (정확 McNemar p ≈ 5×10⁻¹⁶). 사진이 학습 세트에 들어 있을 수 있어 절대
정확도는 낙관적이고, 두 정규화의 **차이**만 의미가 있습니다.

**처음 계획은 틀렸습니다.** 속도계 버튼에서 사진 한 장을 두 범위로 분류해 "확률이 뚜렷하게
높은 쪽"을 고르려 했는데, 틀린 정규화도 쉬운 사진은 대개 맞히고 확률이 더 높게 나오기도
합니다. 1000장에서 이 방법이 맞는 쪽을 고른 비율은 **65%**였습니다. 그래서 결정은 위의
두 근거로 하고, 버튼의 비교 표는 기기에서 두 경로가 모두 동작하는지 보는 확인용으로만
남겼습니다.

---

## 남은 일 / 알려진 한계

- **실기기 숫자가 비어 있습니다.** 위 「측정」 표는 폰에서 속도계 버튼을 눌러야 채워집니다
- **리사이즈 보간.** 앱은 `image` 패키지 기본값인 nearest로 줄입니다. 같은 1000장에서
  bilinear로 바꾸면 top-1이 1.9%p 오릅니다(83.9 → 85.8%). 기기에서 리사이즈 시간이 얼마나
  느는지 재고 바꿀지 정합니다
- **디코드가 남은 병목일 가능성이 큽니다.** 텐서 변환은 줄였지만 `img.decodeImage`는
  여전히 Dart 구현입니다. 측정 결과를 보고 `dart:ui` 디코더나 isolate로 옮길지
  판단할 계획입니다
- **다운로드 실패 처리가 얕습니다.** 네트워크 오류 시 빈 목록을 반환하고 로그만 남깁니다.
  재시도와 사용자 안내가 필요합니다
- **앱 쪽 zip 무결성 검증 없음.** `tools/fetch_model.py`는 SHA-256을 확인하지만, 앱은
  받은 zip을 그대로 풉니다. `/models` 응답에 체크섬을 실어 앱에서도 확인해야 합니다
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
- [**dinov3-image-search**](https://github.com/chaejoon23/dinov3-image-search)
  — 자기지도 임베딩으로 닮은 이미지·장소를 찾는 검색

이 레포가 **배포**, Bin_pind가 **화질**, 위가 **표현학습** 쪽입니다.
