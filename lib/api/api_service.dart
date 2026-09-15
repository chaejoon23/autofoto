// lib/api/model_downloader.dart

import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:archive/archive.dart';

class ApiService {
  /// 모델 서버 주소. 빌드할 때 넘긴다:
  ///   flutter run --profile --dart-define=MODEL_SERVER=http://192.168.0.12:9000
  /// 서버는 `python3 tools/model_server.py`로 맥에서 띄운다 (README 「실행」).
  /// 기본값 localhost는 iOS 시뮬레이터에서만 동작한다.
  static const String _baseUrl =
      String.fromEnvironment('MODEL_SERVER', defaultValue: 'http://localhost:9000');

  // 1. 서버에서 모델 목록을 가져오는 함수
  static Future<List<String>> getAvailableModels() async {
    try {
      final response = await http.get(Uri.parse('$_baseUrl/models'));
      if (response.statusCode == 200) {
        // JSON 응답을 List<String> 으로 변환
        List<dynamic> data = jsonDecode(response.body);
        return data.map((item) => item.toString()).toList();
      }
    } catch (e) {
      print('모델 목록 가져오기 실패: $e');
    }
    return []; // 실패 시 빈 리스트 반환
  }

  // 2. 특정 모델을 다운로드하고 압축을 푸는 함수 (수정됨)
  // 이제 각 모델은 자신의 이름으로 된 폴더 안에 저장됩니다.
  static Future<bool> downloadAndUnzipModel(String modelName) async {
    try {
      final url = Uri.parse('$_baseUrl/download-model?model_name=$modelName');
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final zipBytes = response.bodyBytes;
        final archive = ZipDecoder().decodeBytes(zipBytes);

        final appDir = await getApplicationDocumentsDirectory();
        
        // 각 모델별로 폴더 생성 (예: .../models/mobilenetv2/)
        final modelDir = Directory('${appDir.path}/models/$modelName');
        if (!await modelDir.exists()) {
          await modelDir.create(recursive: true);
        }

        for (final file in archive) {
          final filePath = '${modelDir.path}/${file.name}';
          if (file.isFile) {
            final outFile = File(filePath);
            await outFile.writeAsBytes(file.content as List<int>);
          }
        }
        print('$modelName 모델 다운로드 및 압축 해제 완료');
        return true;
      }
    } catch (e) {
      print('$modelName 모델 다운로드 중 오류: $e');
    }
    return false;
  }
}