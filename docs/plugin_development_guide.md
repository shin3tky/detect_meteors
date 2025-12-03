# プラグイン開発ガイド

`detect_meteors` では `PluginInfo` メタデータと決められたエントリーポイント／モジュール構造を持つ Python モジュールをロードし、検出器・前処理・出力処理を差し替えられます。ここではプラグインを正しく認識させるために必要な要素と、実行時に行われるバリデーションについて説明します。

## PluginInfo の必須フィールド

各プラグイン実装は `plugin_info` 属性に `detect_meteors.app.PluginInfo` のインスタンスを持たせます。必須フィールドは次のとおりです。

- `name` (`str`): レジストリのキー。モジュール内のマッピングキーと一致させます。
- `version` (`str`): プラグインのバージョン文字列。
- `capabilities` (`List[str]`): プラグインの役割や追加機能を示すタグ。`detector`/`preprocessor`/`writer` などの役割名を含め、必要に応じて拡張タグを追加します。

`PluginInfo` は空文字や空のリストを許容せず、`plugin_loader` でも型と内容を再確認します。

## エントリーポイントの設定例

プラグインを Python パッケージとして配布する場合は `detect_meteors.plugins` エントリーポイントにモジュールを登録します。例として `pyproject.toml` では次のように宣言します。

```toml
[project.entry-points."detect_meteors.plugins"]
my-meteors-plugin = "my_package.plugin_module"
```

`setup.cfg` を利用する場合は `[options.entry_points]` に同様のセクションを追加してください。エントリーポイントがモジュールを指していれば、そのモジュール内の実装が自動検出されます。

## モジュール内マッピングの形式

1 つのプラグインモジュールは下記の辞書を任意に定義し、キーをレジストリ名・値を実装インスタンスとして登録します。

```python
DETECTORS = {"custom_detector": CustomDetector()}
PREPROCESSORS = {"custom_pre": CustomPreprocessor()}
OUTPUT_WRITERS = {"custom_writer": CustomWriter()}
```

- 辞書キーは `plugin_info.name` と一致させます。
- 辞書自体は `dict` である必要があります。`None` の場合はスキップされ、`dict` 以外の場合はエラーとして報告されます。

## ライフサイクルフック

各実装に `initialize()` と `shutdown()` を定義すると、登録時・解除時に自動で呼び出されます。重い初期化は `initialize()` に集約し、リソースの解放は `shutdown()` で行うと安全です。

## 実行時バリデーションと警告

`plugin_loader` はロード時に以下を検査し、問題がある場合は警告として `--list-plugins` 出力やログに含めます。

- `plugin_info` が `PluginInfo` 型であること、`name`/`version`/`capabilities` が空でない文字列（および文字列リスト）であること。
- 実装が期待するプロトコルに沿っているか (`inspect.signature` を用いて `preprocess()` や `write()` の必須引数を確認、検出器は `app.Detector` を継承しているか)。

`plugin_info` が欠落・無効な場合はその項目が登録されず、プロトコル警告がある場合でも登録は継続されます。`--list-plugins` を実行すると検出された警告が確認できます。
