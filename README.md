# doc-search-test

tf-idf-vectorizer のテスト用リポジトリですよん。

## 概要

このプロジェクトは Rust 製の tf-idf-vectorizer libの動作検証・テストを目的・ベンチマークとしています。

- Sudachi による日本語トークン化対応
- 類似文書検索
- コーパス・インデックスの保存/読込
- 並列・ストリーム処理による高速インデックス構築 (600doc/sec くらい)

## 使い方

実行ディレクトリにsudachiの設定と辞書を配置してください。
配置例（実行ディレクトリ直下）:

eg. 
```text
.
└─ config/
    ├─ sudachi.json
    ├─ char.def
    ├─ rewrite.def
    ├─ system.dic
    └─ unk.def
```

1. 必須: 文書ディレクトリを用意し、`--docs DIR` で指定してください。
2. Sudachi コマンドが必要な場合は `--sudachi CMD` で指定できます。
3. クエリ検索は `--query "検索文"` または対話モードで実行可能です。

詳細なオプションは `cargo run --release -- -h` で確認できます。

んで

```sh
cargo run --release -- --help
```

# Query format
key_word:
- &: AND
- |: OR
- !: NOT
- []: group

## 注意

- 本リポジトリは tf-idf-vectorizer のテスト・検証用途です。
- 実運用・本番利用は想定していません。
