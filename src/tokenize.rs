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
        const MAX_CHUNK_CHARS: usize = 2000;
        let tokenizer = self.new_tokenizer();

        // 短ければそのまま
        if text.chars().count() <= MAX_CHUNK_CHARS {
            let result = tokenizer.tokenize(text, mode, false)?;
            let mut vec = Vec::new();
            vec.push(result);
            return Ok(Tokenized { result: vec });
        }

        // 区切りに使う記号（改行含む）
        let delimiters = "。．、？！.!?；;,，\n\r";
        let mut aggregated = Vec::new();
        let mut buffer = String::new();
        let mut buffer_len = 0usize;

        for segment in text.split_inclusive(|c| delimiters.contains(c)) {
            let segment_len = segment.chars().count();

            if buffer_len + segment_len > MAX_CHUNK_CHARS {
                if !buffer.is_empty() {
                    let chunk = tokenizer.tokenize(&buffer, mode, false)?;
                    aggregated.push(chunk);
                    buffer.clear();
                    buffer_len = 0;
                }

                if segment_len > MAX_CHUNK_CHARS {
                    let mut temp = String::new();
                    let mut temp_len = 0usize;
                    for ch in segment.chars() {
                        temp.push(ch);
                        temp_len += 1;
                        if temp_len >= MAX_CHUNK_CHARS {
                            let chunk = tokenizer.tokenize(&temp, mode, false)?;
                            aggregated.push(chunk);
                            temp.clear();
                            temp_len = 0;
                        }
                    }
                    if !temp.is_empty() {
                        buffer.push_str(&temp);
                        buffer_len = temp_len;
                    }
                    continue;
                }
            }

            buffer.push_str(segment);
            buffer_len += segment_len;
        }

        if !buffer.is_empty() {
            let chunk = tokenizer.tokenize(&buffer, mode, false)?;
            aggregated.push(chunk);
        }

        Ok(Tokenized { result: aggregated })
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