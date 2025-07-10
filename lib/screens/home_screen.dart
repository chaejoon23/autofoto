import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import '../api/api_service.dart';
import '../services/image_classifier.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImageClassifier _classifier = ImageClassifier();
  final ImagePicker _picker = ImagePicker();

  // 앱 상태를 관리하는 변수들
  String _statusMessage = '앱을 초기화 중입니다...';
  String? _currentModelName;
  bool _isModelReady = false;
  bool _isProcessing = false;

  // 이미지 및 결과 표시를 위한 변수들
  File? _selectedImage;
  List<Map<String, dynamic>>? _classificationResult;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  /// 앱 시작 시 저장된 모델을 불러오는 초기화 함수
  Future<void> _initialize() async {
    final prefs = await SharedPreferences.getInstance();
    // 저장된 모델 이름 불러오기, 없으면 'mobilenetv2'를 기본값으로 사용
    String modelToLoad = prefs.getString('selected_model') ?? 'mobilenetv2';
    await _loadOrDownloadModel(modelToLoad);
  }

  /// 특정 모델을 로드하거나, 없으면 다운로드 후 로드하는 함수
  Future<void> _loadOrDownloadModel(String modelName) async {
    setState(() {
      _currentModelName = modelName;
      _isModelReady = false;
      _isProcessing = true; // 로딩 시작
      _statusMessage = '$modelName 모델을 준비 중입니다...';
      _selectedImage = null;
      _classificationResult = null;
    });

    final appDir = await getApplicationDocumentsDirectory();
    final modelFile = File('${appDir.path}/models/$modelName/model.tflite');

    if (!await modelFile.exists()) {
      setState(() => _statusMessage = '$modelName 모델을 다운로드 중입니다...');
      bool success = await ApiService.downloadAndUnzipModel(modelName);
      if (!success) {
        setState(() {
          _statusMessage = '모델 다운로드 실패';
          _isProcessing = false;
        });
        return;
      }
    }

    await _classifier.initializeClassifier(modelName);
    
    if (_classifier.isModelLoaded()) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('selected_model', modelName); // 성공 시 모델 이름 저장
      setState(() {
        _isModelReady = true;
        _statusMessage = '준비 완료! 사진을 선택하세요.';
      });
    } else {
      setState(() => _statusMessage = '모델 로딩 실패');
    }
    setState(() => _isProcessing = false); // 로딩 끝
  }

  /// 모델 선택 다이얼로그를 보여주는 함수
  Future<void> _showModelSelectionDialog() async {
    List<String> models = await ApiService.getAvailableModels();
    if (!mounted) return;

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('모델 선택'),
          content: SizedBox(
            width: double.maxFinite,
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: models.length,
              itemBuilder: (context, index) {
                final modelName = models[index];
                return ListTile(
                  title: Text(modelName),
                  trailing: _currentModelName == modelName ? const Icon(Icons.check, color: Colors.blue) : null,
                  onTap: () {
                    Navigator.of(context).pop();
                    if (_currentModelName != modelName) {
                      _loadOrDownloadModel(modelName); // 새 모델 선택
                    }
                  },
                );
              },
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('닫기'),
            ),
          ],
        );
      },
    );
  }

  /// 갤러리에서 사진을 선택하고 분류하는 함수
  Future<void> _pickAndClassifyImage() async {
    var status = await Permission.storage.request();
    if (status.isDenied || status.isPermanentlyDenied) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('저장소 접근 권한이 필요합니다.')),
      );
      return;
    }

    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    setState(() {
      _selectedImage = File(image.path);
      _isProcessing = true;
      _statusMessage = '이미지를 분류 중입니다...';
      _classificationResult = null;
    });

    final results = await _classifier.classifyImage(_selectedImage!);
    
    setState(() {
      _classificationResult = results;
      _isProcessing = false;
      _statusMessage = results != null ? '분류 완료!' : '분류 실패';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AutoFoto${_currentModelName != null ? " ($_currentModelName)" : ""}'),
        actions: [
          IconButton(
            icon: const Icon(Icons.swap_horiz_rounded),
            onPressed: _showModelSelectionDialog,
            tooltip: '모델 선택',
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // 이미지 표시 영역
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey.shade300),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: _selectedImage != null
                      ? ClipRRect(
                          borderRadius: BorderRadius.circular(11),
                          child: Image.file(_selectedImage!, fit: BoxFit.cover),
                        )
                      : const Center(child: Text('분류할 이미지를 선택하세요')),
                ),
              ),
              const SizedBox(height: 20),

              // 결과 표시 영역
              SizedBox(
                height: 100, // 결과 표시 영역 높이 고정
                child: _isProcessing
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const CircularProgressIndicator(),
                          const SizedBox(height: 10),
                          Text(_statusMessage),
                        ],
                      )
                    : _classificationResult != null
                        ? Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('분류 결과:', style: Theme.of(context).textTheme.titleMedium),
                              const SizedBox(height: 4),
                              ..._classificationResult!.map((result) {
                                final label = result['label'];
                                final confidence = (result['confidence'] as double) * 100;
                                return Text(
                                  '- $label (${confidence.toStringAsFixed(1)}%)',
                                  style: Theme.of(context).textTheme.bodyLarge,
                                );
                              }).toList(),
                            ],
                          )
                        : Center(child: Text(_statusMessage)),
              ),
              const SizedBox(height: 20),

              // 액션 버튼
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('사진 선택 및 분류'),
                onPressed: _isModelReady && !_isProcessing ? _pickAndClassifyImage : null,
              ),
            ],
          ),
        ),
      ),
    );
  }
}