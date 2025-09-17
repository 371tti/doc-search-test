use std::io::Write;
use std::process::{Command, Stdio};

/// Sudachiの分割モード
#[derive(Clone, Copy, Debug)]
pub enum SudachiMode {
    A,
    B,
    C,
}

impl SudachiMode {
    fn as_str(self) -> &'static str {
        match self {
            SudachiMode::A => "A",
            SudachiMode::B => "B",
            SudachiMode::C => "C",
        }
    }
}

/// 外部コマンド sudachi を使って日本語トークン化する
/// - 入力: &str
/// - 出力: Vec<String>（wakati出力を空白区切りで分割）
/// 失敗時はエラーを返す。
pub fn sudachi_tokenize(input: &str) -> Result<Vec<String>, SudachiError> {
    sudachi_tokenize_with_mode(input, SudachiMode::A)
}

/// 分割モード指定版（表記ゆれをなくし正規化形で返す）
pub fn sudachi_tokenize_with_mode(
    input: &str,
    mode: SudachiMode,
) -> Result<Vec<String>, SudachiError> {
    let mut child = Command::new("sudachi")
        .arg("-a") // 全情報出力
        .arg("-m")
        .arg(mode.as_str())
        .arg("--split-sentences")
        .arg("no")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(SudachiError::Spawn)?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(input.as_bytes()).map_err(SudachiError::Io)?;
    }
    let output = child.wait_with_output().map_err(SudachiError::Io)?;

    if !output.status.success() {
        return Err(SudachiError::Exit(
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let text = String::from_utf8(output.stdout).map_err(SudachiError::Utf8)?;
    let tokens = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with("EOS"))
        .filter_map(|line| line.split('\t').nth(2))
        .map(|s| s.to_string())
        .collect();
    Ok(tokens)
}

#[derive(Debug)]
pub enum SudachiError {
    Spawn(std::io::Error),
    Io(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    Exit(i32, String),
}

impl std::fmt::Display for SudachiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SudachiError::Spawn(e) => write!(f, "failed to spawn sudachi: {}", e),
            SudachiError::Io(e) => write!(f, "io error: {}", e),
            SudachiError::Utf8(e) => write!(f, "utf8 error: {}", e),
            SudachiError::Exit(code, stderr) => {
                write!(f, "sudachi exited with code {}: {}", code, stderr)
            }
        }
    }
}

impl std::error::Error for SudachiError {}

/// 長文を Sudachi に渡すために句読点や記号で分割しつつ最大長を超えないチャンクへ分割する
///
/// 分割トリガ: 。！？!?,、, 改行 等
/// max_len を超えそうな場合は強制切り。UTF-8境界は chars() ベースで安全に。
pub fn split_for_sudachi(text: &str, max_len: usize) -> Vec<String> {
    if text.is_empty() { return Vec::new(); }
    let mut chunks = Vec::new();
    let mut buf = String::with_capacity(max_len.min(4096));
    for ch in text.chars() {
        buf.push(ch);
        let boundary = matches!(ch, '。' | '！' | '!' | '?' | '？' | '、' | ',' | '\n');
        if boundary && buf.len() >= max_len / 2 { // ある程度溜まっていればここで切る
            chunks.push(buf.trim().to_string());
            buf = String::with_capacity(max_len.min(4096));
            continue;
        }
        if buf.len() >= max_len { // 強制切り
            chunks.push(buf.trim().to_string());
            buf = String::with_capacity(max_len.min(4096));
        }
    }
    if !buf.trim().is_empty() { chunks.push(buf.trim().to_string()); }
    chunks
}

/// 長文を安全にトークン化。内部でチャンク分割し連結。
pub fn sudachi_tokenize_large(
    text: &str,
    mode: SudachiMode,
    max_chunk: usize,
) -> Result<Vec<String>, SudachiError> {
    let max_chunk = max_chunk.max(64); // 最低サイズ
    let chunks = split_for_sudachi(text, max_chunk);
    let mut tokens = Vec::new();
    for c in chunks {
        let mut part = sudachi_tokenize_with_mode(&c, mode)?;
        tokens.append(&mut part);
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 外部コマンド依存のため、デフォルトでは無効化
    #[ignore]
    #[test]
    fn test_sudachi_tokenize() {
        let text = "今日は良い天気ですね。";
        let tokens = sudachi_tokenize(text).expect("sudachi tokenize failed");
        println!("{:?}", tokens);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_split_for_sudachi() {
        let long = "これはテストです。これは二文目です！そして三文目です？改行も\n入ります。";
        let parts = split_for_sudachi(long, 20);
        assert!(parts.len() >= 2);
        for p in &parts { assert!(!p.is_empty()); }
    }
}