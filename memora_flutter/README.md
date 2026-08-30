# MEMORA Android client

Flutter client for the MEMORA private-memory API.

## Included

- Android shell with a four-tab Material 3 interface
- Native file picker for documents and images
- Camera capture for receipts, notes, and whiteboards
- Voice-to-text memory queries
- Natural-language memory search with answer and confidence
- Evidence timeline created from returned memories and uploads

## Connect it to the server

1. Start the MEMORA API on the computer.
2. Open **Settings** in the app.
3. Enter the computer's LAN address, for example `http://192.168.1.10:8000`.
4. On an Android emulator, leave the default `http://10.0.2.2:8000`.

The app calls `POST /ingest` for files/photos and `POST /remember` for memory questions.

## Run

```powershell
flutter pub get
flutter run
```

For a debug APK, ensure `JAVA_HOME` points to a JDK (Android Studio's bundled JDK works) and run:

```powershell
flutter build apk --debug
```
