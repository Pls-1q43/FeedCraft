package adapter

import (
	"FeedCraft/internal/util"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/avast/retry-go/v4"
	"github.com/sirupsen/logrus"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/llms/openai"
)

const embeddingCacheTTL = 7 * 24 * time.Hour // 7 days
const maxTextLengthForEmbedding = 8192       // approximate char limit to stay within context window

var (
	embeddingClients sync.Map // cache key -> embeddings.Embedder
)

// getEmbeddingConfig reads embedding-specific env vars with fallback to LLM env vars.
func getEmbeddingConfig() (apiType, apiBase, apiKey, apiModel string) {
	envClient := util.GetEnvClient()
	if envClient == nil {
		log.Fatalf("get env client error.")
	}

	apiType = envClient.GetString("EMBEDDING_API_TYPE")
	if apiType == "" {
		apiType = envClient.GetString("LLM_API_TYPE")
	}
	if apiType == "" {
		apiType = "openai"
	}

	apiBase = envClient.GetString("EMBEDDING_API_BASE")
	if apiBase == "" {
		apiBase = envClient.GetString("LLM_API_BASE")
	}

	apiKey = envClient.GetString("EMBEDDING_API_KEY")
	if apiKey == "" {
		apiKey = envClient.GetString("LLM_API_KEY")
	}

	apiModel = envClient.GetString("EMBEDDING_API_MODEL")
	if apiModel == "" {
		apiModel = envClient.GetString("LLM_API_MODEL")
	}

	return
}

// getOrCreateEmbedder returns a cached or newly created Embedder instance.
func getOrCreateEmbedder() (embeddings.Embedder, string, error) {
	apiType, apiBase, apiKey, apiModel := getEmbeddingConfig()

	if apiModel == "" {
		return nil, "", fmt.Errorf("embedding model not configured: set FC_EMBEDDING_API_MODEL or FC_LLM_API_MODEL")
	}

	cacheKey := fmt.Sprintf("emb|%s|%s|%s|%s", apiType, apiBase, apiKey, apiModel)

	if cached, ok := embeddingClients.Load(cacheKey); ok {
		return cached.(embeddings.Embedder), apiModel, nil
	}

	var client embeddings.EmbedderClient
	var err error

	if apiType == "ollama" {
		if apiBase == "" {
			return nil, "", fmt.Errorf("FC_EMBEDDING_API_BASE must be set when using ollama")
		}
		client, err = ollama.New(
			ollama.WithServerURL(apiBase),
			ollama.WithModel(apiModel),
		)
	} else {
		opts := []openai.Option{
			openai.WithToken(apiKey),
			openai.WithEmbeddingModel(apiModel),
		}
		if apiBase != "" {
			opts = append(opts, openai.WithBaseURL(apiBase))
		}
		client, err = openai.New(opts...)
	}

	if err != nil {
		return nil, "", fmt.Errorf("failed to create embedding client: %w", err)
	}

	embedder, err := embeddings.NewEmbedder(client, embeddings.WithStripNewLines(true))
	if err != nil {
		return nil, "", fmt.Errorf("failed to create embedder: %w", err)
	}

	embeddingClients.Store(cacheKey, embedder)
	logrus.Infof("Embedding client created: type=%s, model=%s", apiType, apiModel)

	return embedder, apiModel, nil
}

// prepareText applies instruction prefix and truncates text.
func prepareText(text, instruction string) string {
	text = strings.TrimSpace(text)
	if len(text) > maxTextLengthForEmbedding {
		text = text[:maxTextLengthForEmbedding]
	}
	if instruction != "" {
		return instruction + ": " + text
	}
	return text
}

// embeddingCacheKey generates a Redis cache key for an embedding vector.
func embeddingCacheKey(prefix, model, instruction, text string) string {
	raw := instruction + "|" + text
	return fmt.Sprintf("embedding_%s_%s_%s", prefix, model, util.GetMD5Hash(raw))
}

// getCachedVector tries to retrieve a cached vector from Redis.
func getCachedVector(key string) ([]float32, bool) {
	cached, err := util.CacheGetString(key)
	if err != nil || cached == "" {
		return nil, false
	}
	var vec []float32
	if err := json.Unmarshal([]byte(cached), &vec); err != nil {
		logrus.Warnf("failed to unmarshal cached embedding for key %s: %v", key, err)
		return nil, false
	}
	return vec, true
}

// cacheVector stores a vector in Redis.
func cacheVector(key string, vec []float32) {
	data, err := json.Marshal(vec)
	if err != nil {
		logrus.Warnf("failed to marshal embedding for caching: %v", err)
		return
	}
	if err := util.CacheSetString(key, string(data), embeddingCacheTTL); err != nil {
		logrus.Warnf("failed to cache embedding: %v", err)
	}
}

// GetEmbedding retrieves the embedding vector for a single text with caching, instruction support, and retry.
func GetEmbedding(ctx context.Context, text string, instruction string) ([]float32, error) {
	embedder, model, err := getOrCreateEmbedder()
	if err != nil {
		return nil, err
	}

	prepared := prepareText(text, instruction)
	cacheKey := embeddingCacheKey("doc", model, instruction, text)

	if vec, ok := getCachedVector(cacheKey); ok {
		return vec, nil
	}

	dispatcher := getLLMDispatcher()

	var result []float32
	result, err = retry.DoWithData(
		func() ([]float32, error) {
			return dispatcher.Execute(ctx, false, func(innerCtx context.Context) ([]float32, error) {
				innerCtx, cancel := context.WithTimeout(innerCtx, 2*time.Minute)
				defer cancel()
				return embedder.EmbedQuery(innerCtx, prepared)
			})
		},
		retry.Attempts(3),
		retry.DelayType(retry.BackOffDelay),
		retry.Delay(1*time.Second),
		retry.MaxDelay(5*time.Second),
		retry.OnRetry(func(n uint, err error) {
			logrus.Warnf("Retrying embedding call (attempt %d): %v", n+1, err)
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("embedding call failed: %w", err)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("embedding API returned empty vector")
	}

	cacheVector(cacheKey, result)
	return result, nil
}

// GetEmbeddings retrieves embedding vectors for multiple texts.
// It checks the cache for each text individually and batches uncached texts.
func GetEmbeddings(ctx context.Context, texts []string, instruction string) ([][]float32, error) {
	embedder, model, err := getOrCreateEmbedder()
	if err != nil {
		return nil, err
	}

	results := make([][]float32, len(texts))
	var uncachedIndices []int
	var uncachedTexts []string

	for i, text := range texts {
		cacheKey := embeddingCacheKey("anchor", model, instruction, text)
		if vec, ok := getCachedVector(cacheKey); ok {
			results[i] = vec
		} else {
			uncachedIndices = append(uncachedIndices, i)
			uncachedTexts = append(uncachedTexts, prepareText(text, instruction))
		}
	}

	if len(uncachedTexts) == 0 {
		return results, nil
	}

	// Batch embed uncached texts
	dispatcher := getLLMDispatcher()
	var vectors [][]float32
	vectors, err = retry.DoWithData(
		func() ([][]float32, error) {
			return dispatcher.Execute(ctx, false, func(innerCtx context.Context) ([][]float32, error) {
				innerCtx, cancel := context.WithTimeout(innerCtx, 2*time.Minute)
				defer cancel()
				return embedder.EmbedDocuments(innerCtx, uncachedTexts)
			})
		},
		retry.Attempts(3),
		retry.DelayType(retry.BackOffDelay),
		retry.Delay(1*time.Second),
		retry.MaxDelay(5*time.Second),
		retry.OnRetry(func(n uint, err error) {
			logrus.Warnf("Retrying batch embedding call (attempt %d): %v", n+1, err)
		}),
	)
	if err != nil {
		return nil, fmt.Errorf("batch embedding call failed: %w", err)
	}

	if len(vectors) != len(uncachedTexts) {
		return nil, fmt.Errorf("embedding API returned %d vectors for %d texts", len(vectors), len(uncachedTexts))
	}

	for j, idx := range uncachedIndices {
		results[idx] = vectors[j]
		cacheKey := embeddingCacheKey("anchor", model, instruction, texts[idx])
		cacheVector(cacheKey, vectors[j])
	}

	return results, nil
}

// CosineSimilarity calculates the cosine similarity between two vectors.
// Returns a value in [-1, 1]. Returns 0 if either vector is zero-length or dimensions mismatch.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	denominator := math.Sqrt(normA) * math.Sqrt(normB)
	if denominator == 0 {
		return 0
	}

	return dotProduct / denominator
}
