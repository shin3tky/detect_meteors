# Outputs関連の実装調査メモ

## 1. `outputs` 周辺の主要クラス/関数
- `BaseOutputHandler` / `DataclassOutputHandler` / `PydanticOutputHandler`: 出力ハンドラの抽象基底と構成別の派生。`save_candidate` / `save_debug_image` などの必須フックやプログレス通知用フックを提供。【F:meteor_core/outputs/base.py†L39-L189】【F:meteor_core/outputs/base.py†L191-L234】
- `FileOutputConfig` / `FileOutputHandler` / `create_file_handler`: デフォルトのファイル出力実装と設定用データクラス、ヘルパーファクトリ。【F:meteor_core/outputs/file_handler.py†L26-L191】
- `OutputHandlerRegistry`: 出力ハンドラの発見・登録・生成を担うレジストリ。`create` / `create_default` で設定の型変換やパス上書きを実行。【F:meteor_core/outputs/registry.py†L33-L189】
- `discover_handlers` / `_discover_handlers_internal`: エントリポイントやプラグインディレクトリからハンドラを探索するユーティリティ。【F:meteor_core/outputs/discovery.py†L42-L239】
- `ProgressManager` / `load_progress` / `save_progress`: 進捗の読み書きと復旧を支援するユーティリティ。【F:meteor_core/outputs/progress.py†L22-L239】
- `OutputWriter`: レガシー `FileOutputHandler` ラッパー（後方互換 API）。【F:meteor_core/outputs/writer.py†L18-L87】

## 2. 例外・ログ出力の抽出
- コンフィグ型不整合: データクラス指定なしや型不一致の場合に `TypeError` を送出（例: `DataclassOutputHandler` で dataclass でない/インスタンス不一致）。【F:meteor_core/outputs/base.py†L214-L234】
- Pydantic 利用不可: Pydantic が無い場合に `ImportError` を送出、型不一致でも `TypeError` を使用。【F:meteor_core/outputs/base.py†L236-L269】
- レジストリ生成失敗: 未定義デフォルトや設定不足に `TypeError`、存在しないフィールドのパス上書きには `AttributeError`、取得失敗には共有基盤の `KeyError` を継承利用。【F:meteor_core/outputs/registry.py†L117-L189】
- プラグイン検証/探索: 条件に合わないハンドラや重複発見時は `warnings.warn` で通知。読み込み失敗も warning で報告し、処理は継続。【F:meteor_core/outputs/discovery.py†L71-L239】
- 進捗ファイル I/O: 読み書き失敗時は `print` ログで通知し `False` / `None` を返す。例外は握りつぶして再スローしない。【F:meteor_core/outputs/progress.py†L75-L193】
- `FileOutputHandler`: 既存ファイルかつ上書き不可なら `False` を返すのみ。明示的な例外送出やロギングはなし。【F:meteor_core/outputs/file_handler.py†L87-L119】

## 3. `inputs` 実装から読み取れるルール（ログ/例外/メッセージ方針）
- コンフィグ検証は `TypeError` と `ValueError` を明確に使い分ける: dataclass/Pydantic でない場合や必須引数不足は `TypeError`、値の妥当性（例: RawLoaderConfig.binning）には `ValueError`。【F:meteor_core/inputs/base.py†L191-L234】【F:meteor_core/inputs/raw.py†L17-L46】
- Pydantic 未導入時は即座に `ImportError`。インスタンス型不一致にも `TypeError` を使用。【F:meteor_core/inputs/base.py†L236-L270】
- レジストリの設定変換 `_coerce_config` は例外原因を `from exc` でラップしてメッセージにプラグイン種別/名前を含める。デフォルト生成失敗は `TypeError`/`ValueError` を段階的に送出。【F:meteor_core/plugin_registry_base.py†L59-L137】
- プラグイン探索は `warnings.warn` で通知しつつ処理継続。重複や無効クラスも warning で知らせ、デプリケーションも warning で扱う。【F:meteor_core/inputs/discovery.py†L50-L239】
- レジストリ生成メソッドは存在しないデフォルト設定時に `TypeError` を送出し、取得できないプラグインは `KeyError` で不正入力を明示する。【F:meteor_core/inputs/registry.py†L80-L118】
- ロガーは使用せず、診断は warnings に集約。ファイル I/O を伴うローダー側では例外をそのまま上位へ伝搬（`RawImageLoader.load` の `Exception` ドキュメント）。【F:meteor_core/inputs/raw.py†L62-L95】

## 4. 差分メモ（不足/過剰/整合性の問題）
- 不足: `FileOutputHandler` や `ProgressManager` では warnings ではなく `print`/無通知でスキップしており、`inputs` 側の一貫した warnings ベースの通知方針に比べて診断手段が弱い。【F:meteor_core/outputs/file_handler.py†L87-L119】【F:meteor_core/outputs/progress.py†L75-L193】
- 過剰: 進捗 I/O で例外を完全に握りつぶして `False` を返すのみのため、`inputs` のような原因付き例外伝搬より情報が減っている。再スローしないことでデバッグ困難。【F:meteor_core/outputs/progress.py†L75-L193】
- 整合性の問題: レジストリやディスカバリの例外/警告スタイルは `inputs` と整合している一方、出力ファイル操作はエラーレベル・メッセージ構造が統一されていない（警告/例外なし）。同様のパス上書き失敗に `AttributeError` を使う点は `inputs` 側に相当箇所がなく、エラーモデルが若干ずれている。【F:meteor_core/outputs/registry.py†L168-L189】【F:meteor_core/inputs/registry.py†L80-L118】
