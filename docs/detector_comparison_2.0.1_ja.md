# 2.0.1 の検出器比較実験計画

領域ベースと線分ベースの試作を、組み込み Hough と同じ隣接 RAW ペアで比較する。比較を通して、現行のプラグイン契約で異なる判定根拠を扱える範囲を確かめ、2.0.1 の参考実装を選ぶ。初回は同梱系列で手順と出力を検証し、検出方式の採否には別の確認済み系列を使う。

## 既存の拡張点

| 処理 | 使う拡張点 | 実験での使い方と制約 |
| --- | --- | --- |
| RAW 読み込みと ROI | 組み込み `raw` ローダー、`PipelineConfig` | `binning: 2`、`normalize: false` を全方式で共通化する。ROI は全画面で固定し、座標はビニング後の画像を基準にする。 |
| 単一画像の補正 | `on_image_loaded` | 画像ごとに完結する補正を試す場合に使える。初回の比較では無効にする。2枚の差分や差分後の DoG・トップハットは検出器内に置く。 |
| ペアの検出 | `DataclassDetector.detect(DetectionContext)` | 現在・直前画像、ROI、フレーム番号を受け取れる。各方式を `detector_name` と `detector_config` で選択する。領域方式は必須の `compute_line_score` を `(0.0, [])` で実装し、領域の根拠を `extras` に載せる。線分方式は既存の `lines` に整数座標を渡し、LSD のサブピクセル座標は `extras` に保持する。 |
| ペアへの注記 | `on_detection_complete` | 画素と検出結果の両方を参照できる。比較の本走行では無効にし、連携確認で `extras` に注記を1つ足す。逐次処理ではメイン、並列処理ではワーカーで動く。 |
| 時系列の注記 | `on_all_detections_sorted`、組み込み `aircraft_trail` | 本走行とは別に、Hough と線分方式で航空機注記を試す。組み込みフックは最長線分を使うため、`lines=[]` の領域方式の比較には時系列注記用の別入力が必要になる。ソート済みフックは画素を受け取らず、再開時はその実行で新たに処理したペアが対象になる。 |
| 比較出力 | `FileOutputHandler` を継承した実験用ハンドラー | `on_detection_result` で候補・非候補の両方をメインプロセスから JSON Lines に記録する。候補 RAW の保存は既存ハンドラーに任せる。この通知は候補保存や全件ソート済みフックに先行するため、保存の成否と最終的な航空機注記は別に照合する。 |

検出器・フック・出力は通常のプラグイン探索で登録する。ワーカーで動く検出器とフックは import 可能なモジュールに置く。`on_file_found` は隣接ペアの構成を変え得るため、本走行では無効にする。`on_batch_results_sorted` は各バッチ内の注記の確認に使え、全系列の連続性は `on_all_detections_sorted` で扱う。`on_output_saved` は候補保存後の通知に使える。

最初は通常の CLI で方式ごとに独立した実行を行い、報告をあとから結合する。Hough の通常結果には詳細な不採用理由が含まれないため、比較レポートでは未収録と記録し、差が出たペアの理由は Hough 専用の `--analyze-image` で別に調べる。出力ハンドラーで再検出を行わず、検出器の処理時間を比較できるようにする。[プラグインの実行順とインスタンス寿命](PLUGIN_AUTHOR_GUIDE.md#hook-execution-and-state) · [登録方法](PLUGIN_AUTHOR_GUIDE.md#35-plugin-discovery) · [画像分析オプション](COMMAND_OPTIONS.md#read-only-image-analysis-v1611)

## 比較する4条件

| 条件 | 処理 | 判定根拠 |
| --- | --- | --- |
| `hough` | 既存の `aircraft_trail_sample.yaml` に記録された7つの Hough パラメータを使う。フックは解除する。 | 現行の候補判定、フレーム全体の線分長合計、Hough 線分。 |
| `region` | ROI 内の16ビット絶対差分を二値化し、8近傍でラベリング。連結成分の面積、長軸と短軸、長軸に沿った局所コントラストを測る。 | 面積・長軸長・縦横比・コントラストの条件を満たす領域が1つでもあれば候補にする。外接矩形、支持された軸区間、各条件の結果を `extras` に記録する。 |
| `lsd_dog` | 絶対差分に DoG を適用し、正の応答を8ビットに写像して LSD に渡す。 | 長さと元の差分上の局所コントラストの条件を満たす線分が1つでもあれば候補にする。各線分の判定根拠も記録する。 |
| `lsd_tophat` | 絶対差分に白トップハットを適用する。8ビット写像、LSD の設定、後段の判定は `lsd_dog` と共通にする。 | `lsd_dog` と同じ測定項目を記録し、前処理の違いを比較する。 |

- 局所コントラストは、支持区間に沿ってサンプリングした中心帯と両側の背景帯の輝度中央値の差で測る。帯の幅と境界の扱いを共通化し、ROI 外の画素を集計から外す。試験用のスコアは候補ごとのコントラストを元の差分の画素値単位で保持する。Hough の線分長合計は別の尺度として扱う。
- 初回の動作確認では、領域の二値化に差分 `> 9` を使い、DoG の標準偏差を `1 px` と `3 px`、白トップハットの構造要素を `9 × 9 px` の楕円とする。LSD の変換は `clip(255 × 正の応答 / 上限, 0, 255)` を `uint8` に量子化する。上限は開発用系列で決めて両条件で共有し、クリップされた割合を記録する。長さ・面積・形状・コントラストの判定閾値と背景帯の幅も、開発用系列で決めて設定ファイルに固定する。
- RANSAC は、領域方式で分断した軌跡の測定が問題になった場合に追加し、追加前後を別条件として比較する。乱数シードと試行上限を記録する。[OpenCV のフィルタ](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html) · [OpenCV LSD](https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html) · [RANSAC による直線当てはめ](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ransac.html)

## 実験手順

1. **入力と注釈を固定する。** `rawfiles/2024GEMINI_AIRCRAFT` のチェックサムを確認する。ファイル名順の12枚から `_C140335.ORF` を初回参照とする11組を作る。確認済み流星は `_C140338.ORF` と `_C140344.ORF` にあり、両方とも航空機と同居する。各流星の軌跡に、ビニング後の中心線と許容幅を目視で付ける。確認できる航空機の軌跡と、判定不明の領域も記録する。絶対差分で次のペアに現れる消失像には直前画像由来の印を付け、流星イベントを二重計上しない。[同梱系列のラベルとチェックサム](../rawfiles/2024GEMINI_AIRCRAFT/README.md)
2. **Hough の対照結果を取る。** 固定パラメータ、全画面 ROI、フックなしで実行する。`--auto-params` は使わず、各方式の実効設定を保存する。同梱系列は既存の Hough 設定が得られた系列なので、試作と手順の確認に使う。別の撮影セッションを確保し、そこでの評価前に設定と照合基準を固定する。
3. **4条件を独立に実行する。** 逐次処理で入力順、ローダー、ROI、デバッグ出力設定をそろえる。条件・繰り返しごとに新しい出力先、進捗ファイル、比較レポートを使う。進捗のハッシュは検出器名や `detector_config` を識別しないため、進捗を条件間で共有すると別方式で処理済みのペアがスキップされ得る。
4. **ペア単位で照合する。** 入力一覧から作った期待ペアと JSON Lines を突き合わせる。読み込みや検出に失敗したペアはログと照合して別に数え、欠損行を非候補として扱わない。幾何の照合は、注釈された区間のうち、向きが近く許容幅内にある検出根拠が覆った割合で行う。暫定基準は向きの差 `15°` 以内、区間の `50%` 以上とし、採点前に固定する。複数の短い線分は同じ軌跡上の区間を合算し、重複部分は一度だけ数える。領域方式は外接矩形の重なりに加え、成分画素が支持する軸区間を使う。
5. **フック・並列・再開を確認する。** 注記フックなしとありで候補判定が一致し、注記が JSON Lines に届くことを確認する。2つの spawn ワーカーと固定バッチサイズ4で再実行し、ペア順に並べた判定と幾何を逐次処理と比較する。中断・再開は同一条件の専用進捗で別に試し、実行ごとの報告と進捗を照合して未処理ペアと再試行を追跡する。Hough と線分方式の航空機注記は、完走後に進捗から確認する。時系列フックの結果は保存済み候補の判定から独立して評価する。

Hough 対照の確認は、リポジトリのルートで次のコマンドを使う。比較レポート用のハンドラーを導入する前にも実行できる。

```bash
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/detect-meteors-hough-XXXXXX")
uv run --locked detect-meteors \
  --config config_examples/aircraft_trail_sample.yaml \
  --detector hough --hooks '' --no-roi --no-resume \
  --workers 1 --batch-size 4 --no-auto-batch-size --profile \
  --output "$run_dir/candidates" --debug-dir "$run_dir/debug" \
  --progress-file "$run_dir/progress.json"
```

## 記録と判定

| 記録 | 内容 |
| --- | --- |
| `run.json` | コミット、OS、Python・OpenCV・NumPy のバージョン、入力のチェックサム、並び順、ROI、検出器名とバージョン、全設定、ワーカー数。出力ハンドラーの設定にも実行専用の保存先を渡す。 |
| `pairs.jsonl` | 現在・直前のファイル名と番号、候補判定、スコア名・値・単位、座標系、線分または領域、利用可能な通過・不通過理由、検出処理の時間。JSON に変換できる値を使い、座標や特徴量の上限を記録する。 |
| 比較結果 | 期待ペア数、処理成功・失敗数、Hough から増減した候補、確認済み流星画像の候補化、注釈した軌跡への一致を集計する。候補化と軌跡一致は別欄にする。Hough では候補判定にどの軌跡が寄与したかを全体スコアから特定できない。 |
| 処理コスト | 同じ環境で条件ごとに逐次3回を目安に測り、検出器内の実行時間と出力を含む総時間を分けて中央値を出す。逐次処理のピーク RSS を同じ計測方法で記録する。候補 RAW 保存枚数も添えて I/O の差を説明できるようにする。 |

参考実装の条件は、各方式で期待したペアの処理と報告が揃い、判定の根拠を人が追え、逐次・並列・再開の動作を確認できること。独立系列では、Hough が候補にした確認済み流星のフレームを維持できるかを先に確認し、軌跡一致数、目視確認枚数、処理コストの順に比較する。撮影条件をまたぐ採否判断には、複数の確認済み系列を使う。

## 領域ベースの試作結果

実験パッケージは、既存のエントリーポイントから [`region` 検出器](../experiments/region_comparison/region_detector.py) と [`pair_report` 出力ハンドラー](../experiments/region_comparison/pair_report.py) を登録する。`pair_report` は通常の候補 RAW 保存に加え、全ての成功ペアの判定と根拠を JSON Lines に書く。

領域方式は絶対差分 `> 9` の画素を8近傍で連結する。24画素以上かつ外接矩形の対角長が15 px以上の成分を計測し、最小二乗で求めた軸への投影長が15 px以上、投影長と横幅の比が3以上、コントラストが15 DN以上なら採用する。コントラストは軸に沿う中央3画素帯と両側6〜8 pxの帯の中央値の差。領域方式のフレームスコアは採用成分の最大コントラストで、根拠には中央帯に実画素がある軸区間を記録する。これらは今回の動作確認用の初期値。

チェックサムが一致した同梱 RAW 12枚から、両方式とも同じ11ペアを処理した。全画面 ROI、固定パラメータ、フックなし、デバッグ画像の保存なしで比較した。確認済み2本の[暫定軌跡](../experiments/region_comparison/meteor_labels.json)は、検出結果を見る前に、対象フレームから直前・直後画像の平均を引いた表示で目視し、ビニング後の中心線と許容幅を記録した。照合には向き15°以内、区間の50%以上の基準を使った。

| 指標 | Hough | 領域ベース |
| --- | ---: | ---: |
| 正常に処理・記録したペア | 11 | 11 |
| 候補画像 | 11 | 3 |
| 確認済み流星の画像が候補になった数 | 2 / 2 | 1 / 2 |
| 候補判定と軌跡の照合がともに成立した数 | 2 / 2 | 1 / 2 |
| 検出器内の時間合計の中央値（逐次3回） | 0.076 s | 3.95 s |
| CLI 全体の経過時間の中央値（逐次3回） | 6.69 s | 10.56 s |
| 子プロセスのピーク RSS の中央値（逐次3回） | 196 MiB | 639 MiB |

計測環境は macOS 27.2.1（arm64）、Python 3.14.10、OpenCV 5.0.0、NumPy 2.5.5。CLI 全体の時間には RAW の読み込み、候補のコピー、比較出力を含む。領域方式は試作時点で各ペア約5,000件の成分を詳細計測している。報告には採用成分を優先して最大128件を残し、残りの件数も保存した。この系列では採用成分が全件報告に入った。

| 流星が写った画像 | Hough の線分被覆 | 領域方式の採用軸の被覆 | 領域方式の判定 |
| --- | ---: | ---: | --- |
| `_C140338.ORF` | 94.8% | 100.0% | 採用。対応領域の縦横比は5.77、コントラストは89.5 DN。 |
| `_C140344.ORF` | 63.4% | 0.0% | 不採用。計測された軸は83.5%を覆ったが、縦横比1.85とコントラスト10.0 DNが条件未満。 |

領域方式の候補は `_C140336.ORF`、`_C140338.ORF`、`_C140339.ORF`。Hough から増えた候補はなく、除外された8枚には確認済み流星の `_C140344.ORF` が含まれる。`_C140339.ORF` の採用軸は直前フレームの流星の消失像に対応したため、新しい流星には数えなかった。この設定では目視対象が減った一方、流星画像の維持と処理コストの両方で Hough の基準に届かなかった。

領域方式は、逐次処理と2つの spawn ワーカーで11ペア分の判定・幾何が一致した。注記用の `on_detection_complete` フックが出力に反映されることと、4ペア処理後に中断を注入し、同じ進捗から残り7ペアを処理できることも確認した。合成画像では差分閾値の境界と ROI 内外の扱いを8ビット・16ビット入力で検証した。

### 領域方式の比較を再実行する

リポジトリのルートから実行する。試作の wheel は実験用ディレクトリにインストールし、この比較を起動するプロセスから読み込む。

```bash
uv sync --locked
run_root=$(mktemp -d "${TMPDIR:-/tmp}/detect-meteors-region-XXXXXX")
uv build --wheel --out-dir "$run_root/dist" experiments/region_comparison
uv pip install --python .venv/bin/python --no-deps \
  --target "$run_root/plugins" "$run_root"/dist/*.whl

for detector in hough region; do
  run_dir="$run_root/$detector"
  mkdir "$run_dir"
  report_config=$(.venv/bin/python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
print(json.dumps({
    "output_folder": str(run / "candidates"),
    "debug_folder": str(run / "debug"),
    "report_path": str(run / "pairs.jsonl"),
}))
PY
  )
  PYTHONPATH="$run_root/plugins" .venv/bin/detect-meteors \
    --config config_examples/aircraft_trail_sample.yaml \
    --detector "$detector" --hooks '' --no-roi --no-resume \
    --workers 1 --batch-size 4 --no-auto-batch-size --profile \
    --output-handler pair_report --output-handler-config "$report_config" \
    --progress-file "$run_dir/progress.json"
done

.venv/bin/python experiments/region_comparison/summarize.py \
  rawfiles/2024GEMINI_AIRCRAFT \
  "$run_root/hough/pairs.jsonl" "$run_root/region/pairs.jsonl"
```
