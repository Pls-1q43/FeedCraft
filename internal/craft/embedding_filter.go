package craft

import (
	"FeedCraft/internal/adapter"
	"context"
	"strconv"
	"strings"

	"github.com/gorilla/feeds"
	"github.com/samber/lo"
	"github.com/samber/lo/parallel"
	"github.com/sirupsen/logrus"
)

// embedding-filter craft implementation
// Filters RSS items based on semantic similarity to user-defined anchor texts using Embedding models.

var embeddingFilterParamTmpl = []ParamTemplate{
	{
		Key:         "anchor_texts",
		Description: "Anchor texts separated by comma. An item matching any anchor above the threshold is considered a hit. Example: `artificial intelligence,deep learning,LLM`",
	},
	{
		Key:         "similarity_threshold",
		Description: "Similarity threshold (0-1). Items exceeding this value are considered hits.",
		Default:     "0.75",
	},
	{
		Key:         "mode",
		Description: "Filter mode: `include` (keep hits) or `exclude` (remove hits)",
		Default:     "include",
	},
	{
		Key:         "scope",
		Description: "Match scope: `title` (title only), `content` (content only), `all` (title + content)",
		Default:     "all",
	},
	{
		Key:         "instruction_for_query",
		Description: "(Optional) Instruction prefix for anchor texts. For instruction-based embedding models (e.g., mxbai-embed-large). Leave empty for models that don't support instructions.",
		Default:     "",
	},
	{
		Key:         "instruction_for_doc",
		Description: "(Optional) Instruction prefix for article text. Leave empty for models that don't support instructions.",
		Default:     "",
	},
}

func embeddingFilterLoadParam(m map[string]string) []CraftOption {
	anchorTextsStr := m["anchor_texts"]
	if anchorTextsStr == "" {
		logrus.Warn("embedding-filter: anchor_texts is empty, skipping")
		return []CraftOption{}
	}

	anchors := parseAnchorTexts(anchorTextsStr)
	if len(anchors) == 0 {
		logrus.Warn("embedding-filter: no valid anchor texts after parsing, skipping")
		return []CraftOption{}
	}

	threshold := 0.75
	if thStr, ok := m["similarity_threshold"]; ok && thStr != "" {
		if parsed, err := strconv.ParseFloat(thStr, 64); err == nil {
			threshold = parsed
		}
	}

	mode := KeywordIncludeMode
	if modeStr, ok := m["mode"]; ok {
		if modeStr == string(KeywordExcludeMode) {
			mode = KeywordExcludeMode
		}
	}

	scope := KeywordMatchAll
	if scopeStr, ok := m["scope"]; ok {
		switch scopeStr {
		case string(KeywordMatchTitle):
			scope = KeywordMatchTitle
		case string(KeywordMatchContent):
			scope = KeywordMatchContent
		default:
			scope = KeywordMatchAll
		}
	}

	instructionForQuery := m["instruction_for_query"]
	instructionForDoc := m["instruction_for_doc"]

	return []CraftOption{
		OptionEmbeddingFilter(anchors, threshold, mode, scope, instructionForQuery, instructionForDoc),
	}
}

// parseAnchorTexts splits and trims the anchor texts string.
func parseAnchorTexts(raw string) []string {
	parts := strings.Split(raw, ",")
	var result []string
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

// OptionEmbeddingFilter returns a CraftOption that filters feed items by embedding similarity.
func OptionEmbeddingFilter(
	anchorTexts []string,
	threshold float64,
	mode KeywordFilterMode,
	scope KeywordMatchScope,
	instructionForQuery string,
	instructionForDoc string,
) CraftOption {
	return func(feed *feeds.Feed, payload ExtraPayload) error {
		items := feed.Items
		if len(items) == 0 {
			return nil
		}

		ctx := context.Background()

		// 1. Get anchor embeddings (batch, cached)
		anchorVectors, err := adapter.GetEmbeddings(ctx, anchorTexts, instructionForQuery)
		if err != nil {
			logrus.Errorf("embedding-filter: failed to get anchor embeddings: %v", err)
			return err
		}

		// 2. For each item, compute similarity concurrently
		matches := parallel.Map(items, func(item *feeds.Item, _ int) bool {
			text := extractTextForScope(item, scope)
			if strings.TrimSpace(text) == "" {
				return false
			}

			itemVec, embErr := adapter.GetEmbedding(ctx, text, instructionForDoc)
			if embErr != nil {
				logrus.Warnf("embedding-filter: failed to get embedding for item [%s]: %v", item.Title, embErr)
				return false
			}

			// Check against all anchor vectors, hit if any exceeds threshold
			for _, anchorVec := range anchorVectors {
				sim := adapter.CosineSimilarity(itemVec, anchorVec)
				if sim >= threshold {
					logrus.Debugf("embedding-filter: item [%s] matched anchor with similarity %.4f >= %.4f", item.Title, sim, threshold)
					return true
				}
			}
			return false
		})

		// 3. Filter based on mode
		feed.Items = lo.Filter(items, func(_ *feeds.Item, index int) bool {
			switch mode {
			case KeywordIncludeMode:
				return matches[index]
			case KeywordExcludeMode:
				return !matches[index]
			default:
				return true
			}
		})

		logrus.Infof("embedding-filter: %d/%d items retained (mode=%s, threshold=%.2f, anchors=%d)",
			len(feed.Items), len(items), mode, threshold, len(anchorTexts))

		return nil
	}
}

// extractTextForScope extracts the relevant text from a feed item based on the scope setting.
func extractTextForScope(item *feeds.Item, scope KeywordMatchScope) string {
	switch scope {
	case KeywordMatchTitle:
		return item.Title
	case KeywordMatchContent:
		content := item.Content
		if content == "" {
			content = item.Description
		}
		return content
	default: // KeywordMatchAll
		content := item.Content
		if content == "" {
			content = item.Description
		}
		return item.Title + " " + content
	}
}
