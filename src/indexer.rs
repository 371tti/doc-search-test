use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;

use indicatif::{HumanBytes, ProgressBar, ProgressDrawTarget, ProgressStyle};
use tf_idf_vectorizer::{Corpus, Hits, SimilarityAlgorithm, TFIDFData, TFIDFVectorizer, TokenFrequency};

use crate::tokenize::SudachiTokenizer;

pub type DynError = Box<dyn std::error::Error + Send + Sync>;

pub struct DocIndexer {
    pub corpus: Arc<Corpus>,
    pub vectorizer: TFIDFVectorizer<u16>,
    pub tokenizer: SudachiTokenizer,
}

struct ProgressReader<R> {
    inner: R,
    pb: ProgressBar,
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.pb.inc(n as u64);
        }
        Ok(n)
    }
}

struct ProgressWriter<W> {
    inner: W,
    pb: ProgressBar,
    written: u64,
}

impl<W: Write> Write for ProgressWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        if n > 0 {
            self.written = self.written.saturating_add(n as u64);
            self.pb.inc(n as u64);
            if self.written % (4 * 1024 * 1024) < n as u64 {
                self.pb.set_message(format!("{}", HumanBytes(self.written)));
            }
        }
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

impl DocIndexer {
    pub fn new() -> Self {
        let corpus = Arc::new(Corpus::new());
        let vectorizer = TFIDFVectorizer::new(corpus.clone());
        let tokenizer = SudachiTokenizer::new().unwrap();
        DocIndexer { corpus, vectorizer, tokenizer }
    }

    pub fn save_to(&self, index_dir: &Path) -> Result<(), DynError> {
        std::fs::create_dir_all(index_dir)?;

        let corpus_file = std::fs::File::create(index_dir.join("corpus.cbor"))?;
        let vector_file = std::fs::File::create(index_dir.join("vector.cbor"))?;

        let pb_corpus = ProgressBar::new_spinner();
        pb_corpus.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_corpus.enable_steady_tick(std::time::Duration::from_millis(120));
        pb_corpus.set_style(
            ProgressStyle::with_template("{spinner:.green} saving corpus... {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        let pb_vector = ProgressBar::new_spinner();
        pb_vector.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_vector.enable_steady_tick(std::time::Duration::from_millis(120));
        pb_vector.set_style(
            ProgressStyle::with_template("{spinner:.green} saving vector... {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        let corpus_writer = BufWriter::new(corpus_file);
        let vector_writer = BufWriter::new(vector_file);

        let mut corpus_writer = ProgressWriter {
            inner: corpus_writer,
            pb: pb_corpus.clone(),
            written: 0,
        };
        let mut vector_writer = ProgressWriter {
            inner: vector_writer,
            pb: pb_vector.clone(),
            written: 0,
        };

        ciborium::into_writer(&*self.corpus, &mut corpus_writer)?;
        corpus_writer.flush()?;
        pb_corpus.finish_with_message(format!("{}", HumanBytes(corpus_writer.written)));

        ciborium::into_writer(&self.vectorizer, &mut vector_writer)?;
        vector_writer.flush()?;
        pb_vector.finish_with_message(format!("{}", HumanBytes(vector_writer.written)));

        Ok(())
    }

    pub fn load_from(index_dir: &Path) -> Result<Self, DynError> {
        let inst = std::time::Instant::now();
        let corpus_path = index_dir.join("corpus.cbor");
        let vector_path = index_dir.join("vector.cbor");

        let corpus_file = std::fs::File::open(&corpus_path)?;
        let vector_file = std::fs::File::open(&vector_path)?;

        let corpus_len = std::fs::metadata(&corpus_path).map(|m| m.len()).unwrap_or(0);
        let vector_len = std::fs::metadata(&vector_path).map(|m| m.len()).unwrap_or(0);

        let pb_corpus = ProgressBar::new(corpus_len);
        pb_corpus.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_corpus.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} loading corpus... {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({percent}%) ETA {eta_precise}",
            )
            .unwrap()
            .progress_chars("##-"),
        );

        let pb_vector = ProgressBar::new(vector_len);
        pb_vector.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
        pb_vector.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} loading vector... {wide_bar:.cyan/blue} {bytes}/{total_bytes} ({percent}%) ETA {eta_precise}",
            )
            .unwrap()
            .progress_chars("##-"),
        );

        let corpus_reader = BufReader::new(corpus_file);
        let vector_reader = BufReader::new(vector_file);

        let corpus_reader = ProgressReader {
            inner: corpus_reader,
            pb: pb_corpus.clone(),
        };
        let vector_reader = ProgressReader {
            inner: vector_reader,
            pb: pb_vector.clone(),
        };

        let corpus: Corpus = ciborium::from_reader(corpus_reader)?;
        pb_corpus.finish();
        let corpus = Arc::new(corpus);

        let vectorizer: TFIDFData<u16> = ciborium::from_reader(vector_reader)?;
        pb_vector.finish();
        println!("Extending vectorizer...");
        let vectorizer = vectorizer.into_tf_idf_vectorizer(corpus.clone());
        let doc_num = vectorizer.doc_num();
        let tokenizer = SudachiTokenizer::new()?;
        let elapsed = inst.elapsed().as_millis();
        println!("{} documents loaded. Done {} ms", doc_num, elapsed);
        Ok(DocIndexer { corpus, vectorizer, tokenizer })
    }

    pub fn query_to_freq(&self, query: &str) -> Result<TokenFrequency, DynError> {
        let tokenized = self.tokenizer.mix_query_tokenizer(query)?;
        let mut freq = TokenFrequency::new();
        freq.add_tokens(&tokenized);
        Ok(freq)
    }

    pub fn search_doc(&mut self, query: &str) -> Result<Hits<String>, DynError> {
        let inst = std::time::Instant::now();
        let query_freq = self.query_to_freq(query)?;
        let mut results = self.vectorizer.similarity(&query_freq, &SimilarityAlgorithm::CosineSimilarity);
        results.sort_by_score();
        let elapsed = inst.elapsed().as_millis();
        println!("Found {} results in {} ms.", results.list.len(), elapsed);
        Ok(results)
    }
}
