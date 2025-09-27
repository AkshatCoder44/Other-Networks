// ---------- 1. Vocabulary ----------
const vocab = {
  word2id: {},
  id2word: {},
  nextId: 1     // reserve 0 for unknown/padding
};

function addWord(word) {
  if (!vocab.word2id[word]) {
    vocab.word2id[word] = vocab.nextId;
    vocab.id2word[vocab.nextId] = word;
    vocab.nextId++;
  }
}

function buildVocabFromText(text) {
  const words = text.toLowerCase().split(/\W+/).filter(Boolean);
  words.forEach(addWord);
}

function encode(text) {
  const words = text.toLowerCase().split(/\W+/).filter(Boolean);
  return words.map(w => vocab.word2id[w] || 0);  // 0 → unknown
}

function decode(ids) {
  return ids.map(id => vocab.id2word[id] || "<UNK>");
}

// ---------- 2. Embedding Table ----------
function createEmbeddingTable(vocabSize, dim = 50) {
  const table = new Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) {
    table[i] = new Array(dim);
    for (let j = 0; j < dim; j++) {
      table[i][j] = Math.random() * 2 - 1;  // random -1 to 1
    }
  }
  return table;
}

// ---------- 3. Example ----------
buildVocabFromText("The quick brown fox jumps over the lazy dog");

const sentence = "The fox is quick";
const ids = encode(sentence);
console.log("IDs:", ids);

const embeddings = createEmbeddingTable(vocab.nextId, 8);
console.log("Embedding for 'fox':", embeddings[vocab.word2id["fox"]]);
console.log("Decoded back:", decode(ids));