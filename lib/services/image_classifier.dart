import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// 모델이 기대하는 입력 정규화 범위.
///
/// MobileNetV2 배포본은 두 종류가 돌아다닌다. Keras `preprocess_input`·TF-Slim
/// 계열은 [-1, 1]을, TF Hub의 일부 TF2 변환본은 [0, 1]을 기대한다. **틀려도 에러가
/// 나지 않고 정확도만 조용히 떨어진다.**
///
/// 이 앱이 쓰는 모델(`tools/fetch_model.py`가 받는 1001-클래스 MobileNetV2)은
/// [-1, 1]로 확정했다. 근거는 세 가지다 (README 「정규화 범위 확인」).
///  1. 모델 메타데이터: NormalizationOptions mean 127.5 / std 127.5
///  2. ImageNet 클래스당 1장(1000장)을 앱과 같은 전처리(nearest 리사이즈)로 분류:
///     top-1 [-1,1] 83.9% vs [0,1] 75.0% (McNemar p ≈ 5e-16)
///  3. 첫 커밋의 labels.txt가 이 모델의 내장 라벨과 1001줄 모두 일치
///
/// 모델을 바꾸면 이 판단도 다시 해야 한다. 한 장의 확률만으로는 판단할 수 없다
/// ([ImageClassifier.compareNormalizations] 참고).
enum InputNormalization {
  /// pixel / 255 → [0, 1]
  zeroToOne,

  /// (pixel - 127.5) / 127.5 → [-1, 1]
  minusOneToOne,
}

/// 한 장을 분류하는 데 걸린 시간 (마이크로초 단위로 재고 밀리초로 보고).
///
/// 온디바이스에서는 추론 시간만 재면 절반만 보는 것이다. 디코드·리사이즈·
/// 텐서 채우기가 추론보다 오래 걸리는 경우가 흔하다.
class ClassificationTimings {
  const ClassificationTimings({
    required this.decodeUs,
    required this.resizeUs,
    required this.tensorFillUs,
    required this.inferenceUs,
    required this.postprocessUs,
  });

  final int decodeUs;
  final int resizeUs;
  final int tensorFillUs;
  final int inferenceUs;
  final int postprocessUs;

  int get preprocessUs => decodeUs + resizeUs + tensorFillUs;
  int get totalUs => preprocessUs + inferenceUs + postprocessUs;

  static String _ms(int us) => (us / 1000).toStringAsFixed(1);

  /// 로그로 바로 붙일 수 있는 한 줄 요약.
  @override
  String toString() =>
      'decode ${_ms(decodeUs)}ms | resize ${_ms(resizeUs)}ms | '
      'tensor ${_ms(tensorFillUs)}ms | infer ${_ms(inferenceUs)}ms | '
      'post ${_ms(postprocessUs)}ms | total ${_ms(totalUs)}ms';

  /// README 「측정」 표에 그대로 붙일 수 있는 마크다운 행.
  String toMarkdownRows() => '| 디코드 | ${_ms(decodeUs)} ms |\n'
      '| 리사이즈 | ${_ms(resizeUs)} ms |\n'
      '| 텐서 변환 | ${_ms(tensorFillUs)} ms |\n'
      '| **추론** | **${_ms(inferenceUs)} ms** |\n'
      '| 후처리 | ${_ms(postprocessUs)} ms |\n'
      '| 전처리 합계 | ${_ms(preprocessUs)} ms |\n'
      '| 전체 | ${_ms(totalUs)} ms |';
}

class Prediction {
  const Prediction({required this.label, required this.confidence});

  final String label;
  final double confidence;

  Map<String, dynamic> toMap() => {'label': label, 'confidence': confidence};
}

class ImageClassifier {
  Interpreter? _interpreter;
  List<String>? _labels;

  /// 입력 정규화 범위. 모델 카드와 맞지 않으면 정확도가 조용히 떨어진다.
  /// 기본 모델 기준 [-1, 1]이 맞다 ([InputNormalization] 문서의 근거 참고).
  InputNormalization normalization = InputNormalization.minusOneToOne;

  /// true면 [Float32List]에 연속으로 쓰고, false면 예전 중첩 List 경로를 쓴다.
  /// 벤치마크에서 전처리 최적화 전/후를 같은 기기에서 비교하기 위해 남겨 둔다.
  bool useFastPreprocess = true;

  /// 마지막 [classifyImage] 호출의 구간별 소요 시간.
  ClassificationTimings? lastTimings;

  /// 마지막 로드 실패 이유. UI에 그대로 띄워 원인을 바로 보이게 한다.
  String? lastError;

  bool isModelLoaded() => _interpreter != null;

  List<String> get labels => _labels ?? const [];

  Future<void> initializeClassifier(String modelName) async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final modelPath = '${appDir.path}/models/$modelName/model.tflite';
      final labelPath = '${appDir.path}/models/$modelName/labels.txt';

      lastError = null;
      final interpreter = await Interpreter.fromFile(File(modelPath));
      interpreter.allocateTensors();

      final labelData = await File(labelPath).readAsString();
      final labels =
          labelData.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();

      final outputLength = interpreter.getOutputTensor(0).shape.last;
      if (outputLength != labels.length) {
        // 라벨 수와 출력 차원이 다르면 인덱스가 밀려 엉뚱한 이름이 붙는다.
        // 조용히 넘기면 "그럭저럭 맞는 것처럼" 보이기 때문에 여기서 끊는다.
        interpreter.close();
        throw StateError(
          '라벨 ${labels.length}개 / 모델 출력 $outputLength개 — 개수가 맞지 않습니다',
        );
      }

      _interpreter = interpreter;
      _labels = labels;
    } catch (e) {
      // 호출부(home_screen)는 isModelLoaded()로 성공 여부를 판단한다. 여기서
      // 던지면 UI 흐름이 끊기므로, 이유만 남기고 실패 상태로 둔다.
      _interpreter = null;
      _labels = null;
      lastError = e.toString();
    }
  }

  /// 상위 [topK]개 분류 결과. 구간별 소요 시간은 [lastTimings]에 남는다.
  Future<List<Map<String, dynamic>>?> classifyImage(File imageFile, {int topK = 3}) async {
    final predictions = await predict(imageFile, topK: topK);
    return predictions?.map((p) => p.toMap()).toList();
  }

  Future<List<Prediction>?> predict(File imageFile, {int topK = 3}) async {
    final interpreter = _interpreter;
    final labels = _labels;
    if (interpreter == null || labels == null) return null;

    final inputTensor = interpreter.getInputTensor(0);
    final shape = inputTensor.shape; // [1, H, W, 3]
    final height = shape[1];
    final width = shape[2];

    final watch = Stopwatch()..start();

    final bytes = await imageFile.readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) return null;
    final decodeUs = watch.elapsedMicroseconds;

    watch.reset();
    final resized = img.copyResize(decoded, width: width, height: height);
    final resizeUs = watch.elapsedMicroseconds;

    // ── 전처리: 픽셀 → 정규화된 float32 ───────────────────────────────────
    watch.reset();
    final int tensorFillUs;
    final bool floatInput = inputTensor.type == TensorType.float32;

    if (floatInput && useFastPreprocess) {
      final buffer = _fillFloat32(resized, width, height);
      tensorFillUs = watch.elapsedMicroseconds;

      watch.reset();
      // Float32List을 그대로 넘기면 안 된다. tflite_flutter의
      // ByteConversionUtils.convertObjectToBytes는 Uint8List/ByteBuffer만
      // 그대로 통과시키고, 그 밖의 List는 원소마다 4바이트 버퍼를 새로 만들어
      // growable List<int>에 addAll 한다 (15만 회). 같은 메모리를 가리키는
      // Uint8List 뷰로 넘기면 변환 없이 memcpy 한 번으로 끝난다.
      inputTensor.setTo(buffer.buffer.asUint8List());
      interpreter.invoke();
    } else {
      // 예전 경로: 중첩 List. 픽셀당 Dart 객체를 거치므로 224²×3 = 15만 번
      // 대입이 일어난다. 비교용으로만 남겨 둔다.
      final nested = _fillNestedList(resized, width, height);
      tensorFillUs = watch.elapsedMicroseconds;

      watch.reset();
      final output = List.filled(labels.length, 0.0).reshape([1, labels.length]);
      interpreter.run(nested, output);
      final scores = (output[0] as List).cast<double>();
      final inferenceUs = watch.elapsedMicroseconds;

      watch.reset();
      final top = _topK(scores, labels, topK);
      lastTimings = ClassificationTimings(
        decodeUs: decodeUs,
        resizeUs: resizeUs,
        tensorFillUs: tensorFillUs,
        inferenceUs: inferenceUs,
        postprocessUs: watch.elapsedMicroseconds,
      );
      return top;
    }

    final inferenceUs = watch.elapsedMicroseconds;

    watch.reset();
    final outputTensor = interpreter.getOutputTensor(0);
    final raw = outputTensor.data;
    final scores = raw.buffer
        .asFloat32List(raw.offsetInBytes, labels.length)
        .map((v) => v.toDouble())
        .toList(growable: false);
    final top = _topK(scores, labels, topK);
    final postprocessUs = watch.elapsedMicroseconds;

    lastTimings = ClassificationTimings(
      decodeUs: decodeUs,
      resizeUs: resizeUs,
      tensorFillUs: tensorFillUs,
      inferenceUs: inferenceUs,
      postprocessUs: postprocessUs,
    );
    return top;
  }

  Float32List _fillFloat32(img.Image image, int width, int height) {
    final buffer = Float32List(width * height * 3);
    final scale = normalization == InputNormalization.zeroToOne ? 1 / 255.0 : 1 / 127.5;
    final shift = normalization == InputNormalization.zeroToOne ? 0.0 : -1.0;

    var i = 0;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final p = image.getPixel(x, y);
        buffer[i++] = p.r * scale + shift;
        buffer[i++] = p.g * scale + shift;
        buffer[i++] = p.b * scale + shift;
      }
    }
    return buffer;
  }

  List<List<List<List<double>>>> _fillNestedList(img.Image image, int width, int height) {
    final scale = normalization == InputNormalization.zeroToOne ? 1 / 255.0 : 1 / 127.5;
    final shift = normalization == InputNormalization.zeroToOne ? 0.0 : -1.0;

    return [
      List.generate(
        height,
        (y) => List.generate(width, (x) {
          final p = image.getPixel(x, y);
          return [p.r * scale + shift, p.g * scale + shift, p.b * scale + shift];
        }),
      ),
    ];
  }

  List<Prediction> _topK(List<double> scores, List<String> labels, int k) {
    final indices = List<int>.generate(scores.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));
    return indices
        .take(k)
        .map((i) => Prediction(label: labels[i], confidence: scores[i]))
        .toList();
  }

  /// 같은 이미지를 [iterations]번 분류해 구간별 **중앙값**을 돌려준다.
  ///
  /// 첫 호출은 지연 초기화·캐시 워밍 때문에 항상 느리므로 [warmup]회는 버린다.
  /// 평균이 아니라 중앙값을 쓰는 이유는 모바일에서 GC·스케줄링 때문에 한두 번
  /// 크게 튀는 값이 섞이기 때문이다.
  Future<ClassificationTimings?> benchmark(
    File imageFile, {
    int iterations = 20,
    int warmup = 3,
  }) async {
    if (!isModelLoaded()) return null;

    for (var i = 0; i < warmup; i++) {
      await predict(imageFile);
    }

    final samples = <ClassificationTimings>[];
    for (var i = 0; i < iterations; i++) {
      await predict(imageFile);
      final t = lastTimings;
      if (t != null) samples.add(t);
    }
    if (samples.isEmpty) return null;

    int median(int Function(ClassificationTimings) pick) {
      final values = samples.map(pick).toList()..sort();
      return values[values.length ~/ 2];
    }

    return ClassificationTimings(
      decodeUs: median((t) => t.decodeUs),
      resizeUs: median((t) => t.resizeUs),
      tensorFillUs: median((t) => t.tensorFillUs),
      inferenceUs: median((t) => t.inferenceUs),
      postprocessUs: median((t) => t.postprocessUs),
    );
  }

  /// 전처리 최적화 전/후를 같은 기기에서 비교한다. README의 before/after 숫자가
  /// 여기서 나온다.
  Future<String> benchmarkPreprocess(File imageFile, {int iterations = 20}) async {
    final previous = useFastPreprocess;

    useFastPreprocess = false;
    final slow = await benchmark(imageFile, iterations: iterations);
    useFastPreprocess = true;
    final fast = await benchmark(imageFile, iterations: iterations);

    useFastPreprocess = previous;
    if (slow == null || fast == null) return '측정 실패 — 모델이 로드되지 않았습니다';

    final ratio = slow.tensorFillUs / (fast.tensorFillUs == 0 ? 1 : fast.tensorFillUs);
    return '| 텐서 변환 방식 | 소요 시간 |\n'
        '|---|---:|\n'
        '| 중첩 List (기존) | ${(slow.tensorFillUs / 1000).toStringAsFixed(1)} ms |\n'
        '| Float32List (현재) | ${(fast.tensorFillUs / 1000).toStringAsFixed(1)} ms |\n'
        '\n'
        '${ratio.toStringAsFixed(1)}배 단축. 추론 시간은 '
        '${(fast.inferenceUs / 1000).toStringAsFixed(1)} ms.';
  }

  /// 두 정규화 범위로 각각 분류해 상위 1개를 비교한다.
  ///
  /// **이 결과 한 장으로 정규화를 고르면 안 된다.** 틀린 정규화도 쉬운 사진은
  /// 대개 맞히고, 확률이 오히려 더 높게 나오기도 한다. 1000장에서 "확률이 높은
  /// 쪽"을 골랐을 때 맞는 정규화([-1,1])를 고른 비율은 65%뿐이었다. 결정은
  /// 모델 메타데이터와 라벨 있는 여러 장의 정확도로 한다
  /// (`tools/check_normalization.py`). 여기서는 기기에서도 두 경로가 모두
  /// 동작하는지 보는 확인용으로만 쓴다.
  Future<String> compareNormalizations(File imageFile) async {
    final previous = normalization;
    final lines = <String>['| 정규화 | 상위 1개 | 확률 |', '|---|---|---:|'];

    for (final mode in InputNormalization.values) {
      normalization = mode;
      final top = await predict(imageFile, topK: 1);
      final best = top?.isNotEmpty == true ? top!.first : null;
      lines.add(
        '| ${mode.name} | ${best?.label ?? '—'} | '
        '${best == null ? '—' : (best.confidence * 100).toStringAsFixed(1)}% |',
      );
    }

    normalization = previous;
    return lines.join('\n');
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
