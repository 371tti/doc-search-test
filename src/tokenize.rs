use std::path::PathBuf;
use std::sync::Arc;

use sudachi::analysis::stateless_tokenizer::StatelessTokenizer;
use sudachi::config::Config;
use sudachi::dic::dictionary::JapaneseDictionary;
use sudachi::prelude::*;
use sudachi::analysis::Tokenize;

/// std io 経由で呼び出すのが遅いので、Sudachiのライブラリを直接使う版
#[derive(Clone)]
pub struct SudachiTokenizer {
    dictionary: Arc<JapaneseDictionary>,
}

impl SudachiTokenizer {
    const SUDACHI_CONFIG: &str = "./config/sudachi.json";

    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = Config::new(Some(PathBuf::from(Self::SUDACHI_CONFIG)), None, None)?;
        let dict = Arc::new(JapaneseDictionary::from_cfg(&config)?);
        Ok(Self { dictionary: dict })
    }

    fn new_tokenizer(&self) -> StatelessTokenizer<Arc<JapaneseDictionary>> {
        StatelessTokenizer::new(Arc::clone(&self.dictionary))
    }

    pub fn tokenize(&self, text: &str, mode: Mode) -> Result<Tokenized, Box<dyn std::error::Error + Send + Sync>> {
        // 文字数じゃなくバイトで切る（UTF-8デコードの全走査を避ける）
        const MAX_CHUNK_BYTES: usize = 16 * 1024; // 例: 16KB（調整してOK）
        const DELIMS: &[u8] = b"\n\r"; // まずは改行だけを境界にすると速い（句読点までやるなら後述）

        let tokenizer = self.new_tokenizer();

        // 短ければそのまま
        if text.len() <= MAX_CHUNK_BYTES {
            let r = tokenizer.tokenize(text, mode, false)?;
            return Ok(Tokenized { result: vec![r] });
        }

        let bytes = text.as_bytes();
        let mut out = Vec::new();

        let mut start = 0usize;
        while start < bytes.len() {
            let mut end = (start + MAX_CHUNK_BYTES).min(bytes.len());

            // UTF-8境界に合わせる
            while end < bytes.len() && !text.is_char_boundary(end) {
                end -= 1;
            }
            if end <= start {
                // 極端に境界が合わないケースの保険
                end = (start + 1).min(bytes.len());
                while end < bytes.len() && !text.is_char_boundary(end) {
                    end += 1;
                }
            }

            // できれば改行まで後退（軽い delimiter）
            let mut cut = end;
            let search_start = start.max(end.saturating_sub(MAX_CHUNK_BYTES));
            for i in (search_start..end).rev() {
                if DELIMS.contains(&bytes[i]) {
                    cut = i + 1;
                    break;
                }
            }

            let chunk = &text[start..cut];
            if !chunk.trim().is_empty() {
                out.push(tokenizer.tokenize(chunk, mode, false)?);
            }

            start = cut;
        }

        Ok(Tokenized { result: out })
    }


    // pub fn pure_doc_tokenizer(&self, text: &str) -> Result<Vec<Box<str>>, Box<dyn std::error::Error + Send + Sync>> {
    //     let c = self.tokenize(text, Mode::C)?;
    //     Ok(c.tokens())
    // }

    pub fn mix_doc_tokenizer(&self, text: &str) -> Result<(Vec<Box<str>>, u64), Box<dyn std::error::Error + Send + Sync>> {
        let c = self.tokenize(text, Mode::C)?;
        let a = self.tokenize(text, Mode::A)?;
        let mut c_tokens = c.tokens();
        let token_sum = c_tokens.len();
        let a_tokens = a.tokens();
        c_tokens.sort();
        let c_tokens_sub: Vec<Box<str>> = a_tokens.iter()
            .filter(|t| !c_tokens.binary_search(*t).is_ok())
            .cloned()
            .collect();
        let a_speech_tokens = a.speech_tokens();
        let synthetic_tokens: Vec<Box<str>> = c_tokens.into_iter()
            .chain(c_tokens_sub.into_iter())
            .chain(a_speech_tokens.into_iter())
            .collect();
        Ok((synthetic_tokens, token_sum as u64))
    }

    // pub fn pure_query_tokenizer(&self, text: &str) -> Result<Vec<Box<str>>, Box<dyn std::error::Error + Send + Sync>> {
    //     let c = self.tokenize(text, Mode::C)?;
    //     Ok(c.normalized_tokens())
    // }

    pub fn mix_query_tokenizer(&self, text: &str) -> Result<Vec<Box<str>>, Box<dyn std::error::Error + Send + Sync>> {
        let c = self.tokenize(text, Mode::C)?;
        let a = self.tokenize(text, Mode::A)?;
        let mut c_tokens = c.tokens();
        let a_tokens = a.tokens();
        c_tokens.sort();
        let c_tokens_sub: Vec<Box<str>> = a_tokens.iter()
            .filter(|t| !c_tokens.binary_search(*t).is_ok())
            .cloned()
            .collect();
        let a_2gram_tokens: Vec<Box<str>> = a_tokens.windows(2)
            .map(|w| format!("{}{}", w[0], w[1]).into_boxed_str())
            .collect();
        let a_speech_tokens = a.speech_tokens();
        let synthetic_tokens: Vec<Box<str>> = c_tokens.into_iter()
            .chain(c_tokens_sub.into_iter())
            .chain(a_speech_tokens.into_iter())
            .chain(a_2gram_tokens.into_iter())
            .collect();
        Ok(synthetic_tokens)
    }
}

pub struct Tokenized {
    result: Vec<MorphemeList<Arc<JapaneseDictionary>>>,
}

impl Tokenized {
    // pub fn normalized_tokens(&self) -> Vec<Box<str>> {
    //     self.result
    //         .iter().flat_map(|m| {
    //             m.iter()
    //                 .map(|s| s.normalized_form().trim_matches(&[' ', '　']).to_string().into_boxed_str())
    //                 .filter(|s| !s.is_empty())
    //                 .collect::<Vec<Box<str>>>()
    //         }).collect::<Vec<Box<str>>>()
    // }

    pub fn tokens(&self) -> Vec<Box<str>> {
        self.result
            .iter().flat_map(|m| {
                m.iter()
                    .map(|s| s.surface().trim_matches(&[' ', '　']).to_string().into_boxed_str())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<Box<str>>>()
            }).collect::<Vec<Box<str>>>()
    }

    pub fn speech_tokens(&self) -> Vec<Box<str>> {
        self.result
            .iter().flat_map(|m| {
                m.iter()
                    .map(|s| s.reading_form().replace("キゴウ", "").trim().to_string().into_boxed_str())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<Box<str>>>()
            }).collect::<Vec<Box<str>>>()
    }
}