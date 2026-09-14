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

<!-- TODO: 실기기에서 재서 채우세요. 온디바이스 비전에서 가장 중요한 숫자입니다.
     image_classifier.dart 의 classifyImage 앞뒤로 Stopwatch 를 걸면 됩니다. -->

| 항목 | 값 |
|---|---|
| 테스트 기기 | TODO (예: iPhone 13, A15) |
| 모델 | MobileNetV2 1.0 224 (TFLite) |
| 전처리 시간 | TODO ms |
| 추론 시간 | TODO ms |
| 모델 파일 크기 | 약 14MB |

전처리와 추론을 따로 재는 게 좋습니다 — 지금 구현에서는 **전처리가 추론보다 느릴
가능성이 큽니다**(아래 참고).

---

## 남은 일 / 알려진 한계

- **서버 주소 하드코딩.** `--dart-define`이나 설정 화면으로 빼야 합니다. 지금은 같은
  핫스팟에 붙어야만 동작합니다
- **전처리가 느립니다.** 현재 224×224×3 픽셀을 Dart 중첩 `List`로 한 픽셀씩 채웁니다
  (`List.generate` 4중첩 + 이중 루프 = 15만 회 대입). `Float32List`에 연속으로 쓰고
  reshape하면 크게 줄어듭니다. 측정부터 하고 고칠 계획입니다
- **정규화 방식 확인 필요.** 지금 `pixel / 255.0`으로 [0,1]에 맞추는데, MobileNetV2
  배포본에 따라 [-1,1]을 기대하는 것도 있습니다. 모델 카드를 확인해 맞지 않으면
  정확도가 조용히 떨어집니다
- **다운로드 실패 처리가 얕습니다.** 네트워크 오류 시 빈 목록을 반환하고 로그만 남깁니다.
  재시도와 사용자 안내가 필요합니다
- **zip 무결성 검증 없음.** 체크섬 확인을 넣어야 합니다
- 양자화 모델(int8) 비교 미실시 — 크기·속도·정확도 트레이드오프를 재보고 싶습니다

---

## 관련 작업

기기에서 돌아가는 비전 모델에 관심이 있어 이어서 만든 것들입니다.

- [**Bin_pind**](https://github.com/chaejoon23/Bin_pind) — 영상 프레임의 화질을 보고
  쓸 수 있는 프레임만 골라내는 파이프라인. 미니 ISP(화이트밸런스·톤·로컬 콘트라스트)로
  저조도 프레임을 복원해서 넘깁니다
- [**dinov3-image-search**](https://github.com/chaejoon23/DINOv3-image-similarity-search-system-TEST)
  — 자기지도 임베딩으로 닮은 이미지·장소를 찾는 검색

이 레포가 **배포**, Bin_pind가 **화질**, 위가 **표현학습** 쪽입니다.
