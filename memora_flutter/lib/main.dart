import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_to_text.dart' as speech;

void main() => runApp(const MemoraApp());

// const _orange = Color(0xffFF6E42);
// const _offWhite = Color(0xffF9F9F9);
// const _teal = Color(0xff004E72);
// const _navy = Color(0xff092634);

const _orange = Color(0xff710014);
const _offWhite = Color(0xffF2F1ED);
const _teal = Color(0xffB38F6F);
const _navy = Color(0xff161616);

// const _orange = Color(0xff677D6A);
// const _offWhite = Color(0xffD6BD98);
// const _teal = Color(0xff40534C);
// const _navy = Color(0xff1A3636);

class MemoraApp extends StatefulWidget {
  const MemoraApp({super.key});

  @override
  State<MemoraApp> createState() => _MemoraAppState();
}

class _MemoraAppState extends State<MemoraApp> {
  final controller = MemoraController();

  @override
  void initState() {
    super.initState();
    controller.loadSettings();
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MEMORA',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: _orange,
          primary: _orange,
          secondary: _teal,
          surface: _offWhite,
          onSurface: _navy,
          onPrimary: _offWhite,
          onSecondary: _offWhite,
        ),
        scaffoldBackgroundColor: _offWhite,
      ),
      home: HomeScreen(controller: controller),
    );
  }
}

class MemoraController extends ChangeNotifier {
  String baseUrl = 'http://10.0.2.2:8000';
  bool loading = false;
  String? error;
  MemoryResult? latestResult;
  final List<TimelineEntry> timeline = [];

  Future<void> loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    baseUrl = prefs.getString('base_url') ?? baseUrl;
    notifyListeners();
  }

  Future<void> saveBaseUrl(String value) async {
    baseUrl = _normalizeUrl(value);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('base_url', baseUrl);
    notifyListeners();
  }

  Future<bool> search(String query) async {
    if (query.trim().isEmpty) return false;
    loading = true;
    error = null;
    notifyListeners();
    try {
      latestResult = await MemoraApi(baseUrl).remember(query.trim());
      _addSearchTimeline(latestResult!);
      return true;
    } on SocketException {
      error = 'Cannot reach MEMORA. Check the server address and Wi-Fi.';
    } on HttpException catch (e) {
      error = e.message;
    } on FormatException {
      error = 'The server returned an unexpected response.';
    } finally {
      loading = false;
      notifyListeners();
    }
    return false;
  }

  Future<UploadResult> upload(File file) async {
    loading = true;
    error = null;
    notifyListeners();
    try {
      final result = await MemoraApi(baseUrl).upload(file);
      timeline.insert(
        0,
        TimelineEntry(
          time: DateTime.now(),
          title: 'Added ${file.uri.pathSegments.last}',
          subtitle: '${result.chunks} memory fragments indexed',
          icon: Icons.add_photo_alternate_outlined,
        ),
      );
      return result;
    } on SocketException {
      throw const MemoraException(
        'Cannot reach MEMORA. Check the server address and Wi-Fi.',
      );
    } on HttpException catch (e) {
      throw MemoraException(e.message);
    } finally {
      loading = false;
      notifyListeners();
    }
  }

  void _addSearchTimeline(MemoryResult result) {
    for (final memory in result.memories.reversed) {
      timeline.insert(
        0,
        TimelineEntry(
          time: DateTime.now(),
          title: memory.title?.isNotEmpty == true
              ? memory.title!
              : 'Related memory',
          subtitle: memory.source ?? 'Evidence found for “${result.query}”',
          icon: memory.retrievalType == 'primary'
              ? Icons.auto_awesome
              : Icons.link,
        ),
      );
    }
  }

  String _normalizeUrl(String value) {
    final trimmed = value.trim().replaceAll(RegExp(r'/$'), '');
    if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
      return trimmed;
    }
    return 'http://$trimmed';
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.controller});
  final MemoraController controller;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int tab = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      SearchPage(
        controller: widget.controller,
        onShowTimeline: () => setState(() => tab = 2),
      ),
      AddMemoryPage(controller: widget.controller),
      TimelinePage(controller: widget.controller),
      SettingsPage(controller: widget.controller),
    ];
    return Scaffold(
      body: SafeArea(child: pages[tab]),
      bottomNavigationBar: NavigationBar(
        selectedIndex: tab,
        onDestinationSelected: (index) => setState(() => tab = index),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.search), label: 'Remember'),
          NavigationDestination(
            icon: Icon(Icons.add_circle_outline),
            label: 'Add',
          ),
          NavigationDestination(icon: Icon(Icons.timeline), label: 'Timeline'),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}

class SearchPage extends StatefulWidget {
  const SearchPage({
    super.key,
    required this.controller,
    required this.onShowTimeline,
  });
  final MemoraController controller;
  final VoidCallback onShowTimeline;

  @override
  State<SearchPage> createState() => _SearchPageState();
}

class _SearchPageState extends State<SearchPage> {
  final query = TextEditingController();
  final voice = speech.SpeechToText();
  bool listening = false;

  @override
  void dispose() {
    query.dispose();
    super.dispose();
  }

  Future<void> _listen() async {
    if (listening) {
      await voice.stop();
      setState(() => listening = false);
      return;
    }
    final available = await voice.initialize(
      onStatus: (status) {
        if (status == 'done' || status == 'notListening') {
          setState(() => listening = false);
        }
      },
      onError: (_) => setState(() => listening = false),
    );
    if (!available || !mounted) return;
    setState(() => listening = true);
    await voice.listen(
      onResult: (result) => setState(() => query.text = result.recognizedWords),
      listenOptions: speech.SpeechListenOptions(
        listenFor: const Duration(seconds: 30),
      ),
    );
  }

  Future<void> _search() async {
    final success = await widget.controller.search(query.text);
    if (!mounted || success) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(widget.controller.error ?? 'Search failed.')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) {
        final result = widget.controller.latestResult;
        return ListView(
          padding: const EdgeInsets.fromLTRB(20, 26, 20, 32),
          children: [
            const _BrandHeader(),
            const SizedBox(height: 30),
            Text(
              'What do you remember?',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            const Text(
              'Describe it the way you remember it — not the filename.',
            ),
            const SizedBox(height: 18),
            TextField(
              controller: query,
              minLines: 2,
              maxLines: 4,
              textInputAction: TextInputAction.search,
              onSubmitted: (_) => _search(),
              decoration: InputDecoration(
                hintText: '“That receipt for the laptop around Diwali…”',
                filled: true,
                fillColor: Colors.white,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(18),
                  borderSide: BorderSide.none,
                ),
                suffixIcon: IconButton(
                  tooltip: listening ? 'Stop listening' : 'Ask by voice',
                  icon: Icon(
                    listening ? Icons.mic : Icons.mic_none,
                    color: _orange,
                  ),
                  onPressed: _listen,
                ),
              ),
            ),
            const SizedBox(height: 12),
            FilledButton.icon(
              onPressed: widget.controller.loading ? null : _search,
              icon: widget.controller.loading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.auto_awesome),
              label: Text(
                widget.controller.loading
                    ? 'Remembering…'
                    : 'Search your memory',
              ),
            ),
            if (result != null) ...[
              const SizedBox(height: 28),
              _AnswerCard(
                result: result,
                onShowTimeline: widget.onShowTimeline,
              ),
            ] else ...[
              const SizedBox(height: 32),
              const _SuggestionCard(),
            ],
          ],
        );
      },
    );
  }
}

class AddMemoryPage extends StatefulWidget {
  const AddMemoryPage({super.key, required this.controller});
  final MemoraController controller;

  @override
  State<AddMemoryPage> createState() => _AddMemoryPageState();
}

class _AddMemoryPageState extends State<AddMemoryPage> {
  final picker = ImagePicker();

  Future<void> _upload(File file) async {
    try {
      final result = await widget.controller.upload(file);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Memory saved — ${result.chunks} fragments indexed.'),
        ),
      );
    } on MemoraException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(e.message)));
      }
    }
  }

  Future<void> _pickFile() async {
    final selection = await FilePicker.pickFile();
    final path = selection?.path;
    if (path != null) await _upload(File(path));
  }

  Future<void> _capture() async {
    final photo = await picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 88,
    );
    if (photo != null) await _upload(File(photo.path));
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: widget.controller,
      builder: (context, _) => Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const _BrandHeader(compact: true),
            const SizedBox(height: 38),
            Text(
              'Add a memory',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 8),
            const Text(
              'MEMORA can understand documents, photos, and screenshots.',
            ),
            const SizedBox(height: 30),
            _AddOption(
              icon: Icons.folder_open_outlined,
              title: 'Choose a file',
              subtitle: 'PDF, note, image, audio, or document',
              onTap: widget.controller.loading ? null : _pickFile,
            ),
            const SizedBox(height: 16),
            _AddOption(
              icon: Icons.camera_alt_outlined,
              title: 'Capture with camera',
              subtitle: 'Photograph a receipt, note, or whiteboard',
              onTap: widget.controller.loading ? null : _capture,
            ),
            if (widget.controller.loading) ...[
              const SizedBox(height: 28),
              const Center(child: CircularProgressIndicator()),
              const SizedBox(height: 12),
              const Center(
                child: Text('Turning this into a searchable memory…'),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class TimelinePage extends StatelessWidget {
  const TimelinePage({super.key, required this.controller});
  final MemoraController controller;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) => ListView(
        padding: const EdgeInsets.fromLTRB(20, 26, 20, 32),
        children: [
          const _BrandHeader(compact: true),
          const SizedBox(height: 28),
          Text(
            'Memory timeline',
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          const SizedBox(height: 8),
          const Text('Your trail of supporting evidence.'),
          const SizedBox(height: 28),
          if (controller.timeline.isEmpty)
            const _EmptyTimeline()
          else
            ...controller.timeline.map((entry) => _TimelineItem(entry: entry)),
        ],
      ),
    );
  }
}

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key, required this.controller});
  final MemoraController controller;
  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late final TextEditingController url;
  @override
  void initState() {
    super.initState();
    url = TextEditingController(text: widget.controller.baseUrl);
  }

  @override
  void dispose() {
    url.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: ListView(
        children: [
          const _BrandHeader(compact: true),
          const SizedBox(height: 36),
          Text('Connection', style: Theme.of(context).textTheme.headlineMedium),
          const SizedBox(height: 8),
          const Text(
            'Use your computer’s Wi-Fi address when testing on a physical phone.',
          ),
          const SizedBox(height: 24),
          TextField(
            controller: url,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              labelText: 'MEMORA server address',
              hintText: 'http://192.168.1.10:8000',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          FilledButton(
            onPressed: () async {
              final messenger = ScaffoldMessenger.of(context);
              await widget.controller.saveBaseUrl(url.text);
              if (!mounted) return;
              messenger.showSnackBar(
                const SnackBar(content: Text('Server address saved.')),
              );
            },
            child: const Text('Save address'),
          ),
          const SizedBox(height: 24),
          const ListTile(
            leading: Icon(Icons.shield_outlined),
            title: Text('Private by design'),
            subtitle: Text(
              'This app sends memories only to the MEMORA server you choose.',
            ),
          ),
        ],
      ),
    );
  }
}

class _BrandHeader extends StatelessWidget {
  const _BrandHeader({this.compact = false});
  final bool compact;
  @override
  Widget build(BuildContext context) => Row(
        children: [
          Container(
            width: compact ? 34 : 42,
            height: compact ? 34 : 42,
            decoration: BoxDecoration(
              color: _orange,
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.auto_awesome, color: Colors.white),
          ),
          const SizedBox(width: 10),
          Text(
            'MEMORA',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w800,
                  letterSpacing: 1.5,
                ),
          ),
        ],
      );
}

class _AnswerCard extends StatelessWidget {
  const _AnswerCard({required this.result, required this.onShowTimeline});
  final MemoryResult result;
  final VoidCallback onShowTimeline;
  @override
  Widget build(BuildContext context) => Card(
        elevation: 0,
        color: const Color(0x1A004E72),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.psychology_alt_outlined, color: _teal),
                  const SizedBox(width: 8),
                  Text(
                    'Here’s what I found',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ],
              ),
              const SizedBox(height: 14),
              Text(result.answer, style: Theme.of(context).textTheme.bodyLarge),
              const SizedBox(height: 16),
              Text(
                '${(result.confidence * 100).round()}% memory match',
                style:
                    const TextStyle(color: _teal, fontWeight: FontWeight.w700),
              ),
              const Divider(height: 28),
              Text('Evidence', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 8),
              ...result.memories.take(3).map(
                    (m) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Row(
                        children: [
                          const Icon(Icons.description_outlined, size: 18),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              m.title ?? m.source ?? 'Related memory',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              TextButton.icon(
                onPressed: onShowTimeline,
                icon: const Icon(Icons.timeline),
                label: const Text('See evidence timeline'),
              ),
            ],
          ),
        ),
      );
}

class _SuggestionCard extends StatelessWidget {
  const _SuggestionCard();
  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Try a memory, not a keyword',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              const Text(
                '“I saw an image of a circuit with three sensors a few months ago.”',
              ),
            ],
          ),
        ),
      );
}

class _AddOption extends StatelessWidget {
  const _AddOption({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback? onTap;
  @override
  Widget build(BuildContext context) => Card(
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 25,
                  backgroundColor: const Color(0x1A004E72),
                  child: Icon(icon, color: _teal),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(title,
                          style: Theme.of(context).textTheme.titleMedium),
                      const SizedBox(height: 4),
                      Text(subtitle),
                    ],
                  ),
                ),
                const Icon(Icons.chevron_right),
              ],
            ),
          ),
        ),
      );
}

class _EmptyTimeline extends StatelessWidget {
  const _EmptyTimeline();
  @override
  Widget build(BuildContext context) => Card(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              const Icon(Icons.timeline, size: 48, color: _teal),
              const SizedBox(height: 12),
              Text(
                'Your evidence trail will appear here',
                style: Theme.of(context).textTheme.titleMedium,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 6),
              const Text(
                'Search for a memory or add a file to start.',
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
}

class _TimelineItem extends StatelessWidget {
  const _TimelineItem({required this.entry});
  final TimelineEntry entry;
  @override
  Widget build(BuildContext context) => IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(
              width: 42,
              child: Column(
                children: [
                  CircleAvatar(
                    radius: 18,
                    backgroundColor: const Color(0x1A004E72),
                    child: Icon(entry.icon, color: _teal, size: 19),
                  ),
                  Expanded(
                    child: Container(width: 2, color: const Color(0x33004E72)),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.only(bottom: 22),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _formatDate(entry.time),
                      style: Theme.of(context).textTheme.labelMedium,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      entry.title,
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                    const SizedBox(height: 3),
                    Text(entry.subtitle),
                  ],
                ),
              ),
            ),
          ],
        ),
      );
}

String _formatDate(DateTime time) {
  final hour =
      time.hour == 0 ? 12 : (time.hour > 12 ? time.hour - 12 : time.hour);
  final minute = time.minute.toString().padLeft(2, '0');
  final suffix = time.hour >= 12 ? 'PM' : 'AM';
  return 'Just now · $hour:$minute $suffix';
}

class TimelineEntry {
  TimelineEntry({
    required this.time,
    required this.title,
    required this.subtitle,
    required this.icon,
  });
  final DateTime time;
  final String title;
  final String subtitle;
  final IconData icon;
}

class MemoryResult {
  MemoryResult({
    required this.query,
    required this.answer,
    required this.confidence,
    required this.memories,
  });
  final String query;
  final String answer;
  final double confidence;
  final List<MemoryEvidence> memories;
  factory MemoryResult.fromJson(Map<String, dynamic> json) => MemoryResult(
        query: json['query'] as String? ?? '',
        answer: json['answer'] as String? ?? 'I found related memories.',
        confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
        memories: ((json['memories'] as List?) ?? [])
            .whereType<Map>()
            .map((m) => MemoryEvidence.fromJson(Map<String, dynamic>.from(m)))
            .toList(),
      );
}

class MemoryEvidence {
  MemoryEvidence({this.title, this.source, this.retrievalType});
  final String? title;
  final String? source;
  final String? retrievalType;
  factory MemoryEvidence.fromJson(Map<String, dynamic> json) => MemoryEvidence(
        title: json['title'] as String?,
        source: json['source'] as String?,
        retrievalType: json['retrieval_type'] as String?,
      );
}

class UploadResult {
  UploadResult(this.chunks);
  final int chunks;
  factory UploadResult.fromJson(Map<String, dynamic> json) =>
      UploadResult((json['chunks'] as num?)?.toInt() ?? 0);
}

class MemoraException implements Exception {
  const MemoraException(this.message);
  final String message;
}

class MemoraApi {
  MemoraApi(this.baseUrl);
  final String baseUrl;
  Future<MemoryResult> remember(String query) async {
    final request = await HttpClient()
        .postUrl(Uri.parse('$baseUrl/remember'))
        .timeout(const Duration(seconds: 60));
    request.headers.contentType = ContentType.json;
    request.write(jsonEncode({'query': query}));
    final response = await request.close().timeout(const Duration(seconds: 90));
    final body = await utf8.decoder.bind(response).join();
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(_errorMessage(body));
    }
    return MemoryResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  Future<UploadResult> upload(File file) async {
    final boundary = 'memora-${DateTime.now().microsecondsSinceEpoch}';
    final request = await HttpClient()
        .postUrl(Uri.parse('$baseUrl/ingest'))
        .timeout(const Duration(seconds: 30));
    request.headers.set(
      HttpHeaders.contentTypeHeader,
      'multipart/form-data; boundary=$boundary',
    );
    final filename = file.uri.pathSegments.last;
    request.write(
      '--$boundary\r\nContent-Disposition: form-data; name="file"; filename="$filename"\r\nContent-Type: application/octet-stream\r\n\r\n',
    );
    await request.addStream(file.openRead());
    request.write('\r\n--$boundary--\r\n');
    final response = await request.close().timeout(const Duration(minutes: 5));
    final body = await utf8.decoder.bind(response).join();
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(_errorMessage(body));
    }
    return UploadResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
  }

  String _errorMessage(String body) {
    try {
      return (jsonDecode(body) as Map<String, dynamic>)['detail'] as String? ??
          'The server could not process this request.';
    } catch (_) {
      return 'The server could not process this request.';
    }
  }
}
