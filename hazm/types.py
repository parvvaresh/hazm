type Token = str
type Tag = str
type TaggedToken = tuple[Token, Tag]
type Sentence = list[Token]
type TaggedSentence = list[TaggedToken]

type IOBTag = str  # B-NP, I-VP, etc.
type ChunkedToken = tuple[Token, Tag, IOBTag]
type ChunkedSentence = list[ChunkedToken]
