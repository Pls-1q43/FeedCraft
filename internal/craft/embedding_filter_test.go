package craft

import (
	"FeedCraft/internal/adapter"
	"math"
	"testing"

	"github.com/gorilla/feeds"
	"github.com/stretchr/testify/assert"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float64
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
		},
		{
			name:     "similar vectors",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2, 3.1},
			expected: 0.9999, // very close to 1
		},
		{
			name:     "empty vectors",
			a:        []float32{},
			b:        []float32{},
			expected: 0.0,
		},
		{
			name:     "dimension mismatch",
			a:        []float32{1, 2},
			b:        []float32{1, 2, 3},
			expected: 0.0,
		},
		{
			name:     "zero vector",
			a:        []float32{0, 0, 0},
			b:        []float32{1, 2, 3},
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := adapter.CosineSimilarity(tt.a, tt.b)
			if tt.expected == 0.0 || tt.expected == 1.0 || tt.expected == -1.0 {
				assert.InDelta(t, tt.expected, result, 1e-6)
			} else {
				assert.True(t, result > tt.expected-0.01, "expected > %f, got %f", tt.expected, result)
			}
		})
	}
}

func TestCosineSimilarityMathProperties(t *testing.T) {
	a := []float32{3, 4}
	b := []float32{4, 3}

	sim := adapter.CosineSimilarity(a, b)
	// cos(theta) = (3*4 + 4*3) / (5 * 5) = 24/25 = 0.96
	assert.InDelta(t, 0.96, sim, 1e-6)

	// Self-similarity should be 1
	selfSim := adapter.CosineSimilarity(a, a)
	assert.InDelta(t, 1.0, selfSim, 1e-6)

	// Symmetry
	simAB := adapter.CosineSimilarity(a, b)
	simBA := adapter.CosineSimilarity(b, a)
	assert.InDelta(t, simAB, simBA, 1e-9)
}

func TestCosineSimilarityNormalized(t *testing.T) {
	// Normalized vectors
	a := []float32{1.0 / float32(math.Sqrt(2)), 1.0 / float32(math.Sqrt(2))}
	b := []float32{1, 0}

	sim := adapter.CosineSimilarity(a, b)
	// cos(45°) ≈ 0.7071
	assert.InDelta(t, 1.0/math.Sqrt(2), sim, 1e-4)
}

func TestParseAnchorTexts(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "simple",
			input:    "AI,ML,DL",
			expected: []string{"AI", "ML", "DL"},
		},
		{
			name:     "with spaces",
			input:    " AI , machine learning , deep learning ",
			expected: []string{"AI", "machine learning", "deep learning"},
		},
		{
			name:     "empty entries",
			input:    "AI,,ML,,",
			expected: []string{"AI", "ML"},
		},
		{
			name:     "single entry",
			input:    "artificial intelligence",
			expected: []string{"artificial intelligence"},
		},
		{
			name:     "empty string",
			input:    "",
			expected: nil,
		},
		{
			name:     "all commas",
			input:    ",,,",
			expected: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseAnchorTexts(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestExtractTextForScope(t *testing.T) {
	item := &feeds.Item{
		Title:       "Test Title",
		Content:     "Test Content Body",
		Description: "Test Description",
	}

	t.Run("title only", func(t *testing.T) {
		text := extractTextForScope(item, KeywordMatchTitle)
		assert.Equal(t, "Test Title", text)
	})

	t.Run("content only", func(t *testing.T) {
		text := extractTextForScope(item, KeywordMatchContent)
		assert.Equal(t, "Test Content Body", text)
	})

	t.Run("content fallback to description", func(t *testing.T) {
		itemNoContent := &feeds.Item{
			Title:       "Title",
			Content:     "",
			Description: "Fallback Description",
		}
		text := extractTextForScope(itemNoContent, KeywordMatchContent)
		assert.Equal(t, "Fallback Description", text)
	})

	t.Run("all scope", func(t *testing.T) {
		text := extractTextForScope(item, KeywordMatchAll)
		assert.Equal(t, "Test Title Test Content Body", text)
	})
}

func TestEmbeddingFilterLoadParam(t *testing.T) {
	t.Run("empty anchor_texts returns no options", func(t *testing.T) {
		params := map[string]string{
			"anchor_texts": "",
		}
		opts := embeddingFilterLoadParam(params)
		assert.Empty(t, opts)
	})

	t.Run("valid params returns one option", func(t *testing.T) {
		params := map[string]string{
			"anchor_texts":          "AI,ML",
			"similarity_threshold":  "0.8",
			"mode":                  "exclude",
			"scope":                 "title",
			"instruction_for_query": "Represent the query for retrieval",
			"instruction_for_doc":   "Represent the document for retrieval",
		}
		opts := embeddingFilterLoadParam(params)
		assert.Len(t, opts, 1)
	})

	t.Run("defaults applied", func(t *testing.T) {
		params := map[string]string{
			"anchor_texts": "test",
		}
		opts := embeddingFilterLoadParam(params)
		assert.Len(t, opts, 1)
	})

	t.Run("invalid threshold uses default", func(t *testing.T) {
		params := map[string]string{
			"anchor_texts":         "test",
			"similarity_threshold": "not_a_number",
		}
		opts := embeddingFilterLoadParam(params)
		assert.Len(t, opts, 1)
	})
}

func TestEmbeddingFilterTemplateRegistration(t *testing.T) {
	templates := GetSysCraftTemplateDict()
	tmpl, exists := templates["embedding-filter"]
	assert.True(t, exists, "embedding-filter template should be registered")
	assert.Equal(t, "embedding-filter", tmpl.Name)
	assert.NotEmpty(t, tmpl.ParamTemplateDefine)
	assert.NotNil(t, tmpl.OptionFunc)

	// Verify param templates
	paramKeys := make([]string, len(tmpl.ParamTemplateDefine))
	for i, p := range tmpl.ParamTemplateDefine {
		paramKeys[i] = p.Key
	}
	assert.Contains(t, paramKeys, "anchor_texts")
	assert.Contains(t, paramKeys, "similarity_threshold")
	assert.Contains(t, paramKeys, "mode")
	assert.Contains(t, paramKeys, "scope")
	assert.Contains(t, paramKeys, "instruction_for_query")
	assert.Contains(t, paramKeys, "instruction_for_doc")
}
