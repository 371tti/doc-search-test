# doc-search-test

tf-idf-vectorizer のテスト用リポジトリですよん。

## 概要

このプロジェクトは Rust 製の tf-idf-vectorizer libの動作検証・テストを目的・ベンチマークとしています。

- Sudachi による日本語トークン化対応
- TF-IDF/BM25 による類似文書検索
- コーパス・インデックスの保存/読込
- 並列・ストリーム処理による高速インデックス構築

## 使い方

sudachi cli をインストールしといて辞書もぶっこんどいてください。

1. 必須: 文書ディレクトリを用意し、`--docs DIR` で指定してください。
2. Sudachi コマンドが必要な場合は `--sudachi CMD` で指定できます。
3. クエリ検索は `--query "検索文"` または対話モードで実行可能です。

詳細なオプションは `cargo run --release -- -h` で確認できます。

## 例

```sh
cargo run --release -- --docs ./data/ex_docs --query "検索したい文章"
```

## 注意

- 本リポジトリは tf-idf-vectorizer のテスト・検証用途です。
- 実運用・本番利用は想定していません。
