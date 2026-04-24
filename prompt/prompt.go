// Package prompt provides message conversion utilities.
// This mirrors the ai-sdk prompt module.
// Source: ai-sdk/packages/ai/src/prompt/convert-to-language-model-prompt.ts
package prompt

import (
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/message"
)

// StandardizedPrompt represents a validated and standardized prompt.
// Source: ai-sdk/packages/ai/src/prompt/standardize-prompt.ts
type StandardizedPrompt struct {
	// System is the system message(s).
	System any // string | SystemMessage | []SystemMessage

	// Messages are the conversation messages.
	Messages []message.Message
}

// SystemMessage represents a system message with provider options.
type SystemMessage struct {
	Content         string
	ProviderOptions map[string]any
}

// DownloadedAsset represents a downloaded file/image.
type DownloadedAsset struct {
	Data      []byte
	MediaType string
}

// DownloadPlan represents a planned download.
type DownloadPlan struct {
	URL                   *url.URL
	IsURLSupportedByModel bool
}

// DownloadFunc is a function that downloads files.
// It returns nil for entries that couldn't be downloaded.
type DownloadFunc func(ctx context.Context, plans []DownloadPlan) []*DownloadedAsset

// SupportedURLs maps media types to patterns of supported URLs.
// Use "*" as key for all media types.
type SupportedURLs map[string][]*regexp.Regexp

// ConvertOptions are options for ConvertToLanguageModelPrompt.
type ConvertOptions struct {
	// Prompt is the standardized prompt to convert.
	Prompt StandardizedPrompt

	// SupportedURLs maps media types to patterns of supported URLs.
	SupportedURLs SupportedURLs

	// Download is the function to download assets.
	// If nil, defaults to DefaultDownload.
	Download DownloadFunc
}

// LanguageModelMessage is a message in language model format.
// This is the format providers expect.
type LanguageModelMessage struct {
	Role            string
	Content         any // string or []LanguageModelPart
	ProviderOptions map[string]any
}

// LanguageModelPart is a part of a language model message.
type LanguageModelPart struct {
	Type string

	// For text parts
	Text string

	// For file parts
	Data      any // []byte | string (base64) | *url.URL
	MediaType string
	Filename  string

	// For reasoning parts
	// Text is reused

	// For tool-call parts
	ToolCallID       string
	ToolName         string
	Input            any
	ProviderExecuted bool

	// For tool-result parts
	// ToolCallID and ToolName are reused
	Output any

	// For tool-approval-response parts
	ApprovalID string
	Approved   bool
	Reason     string

	// Provider options
	ProviderOptions map[string]any
}

// ConvertToLanguageModelPrompt converts a standardized prompt to language model format.
// This handles:
// - Downloading images/files from URLs when the model doesn't support them
// - Combining consecutive tool messages
// - Validating all tool calls have matching results
func ConvertToLanguageModelPrompt(ctx context.Context, opts ConvertOptions) ([]LanguageModelMessage, error) {
	download := opts.Download
	if download == nil {
		download = DefaultDownload
	}

	// Download assets for URLs that the model doesn't support
	downloadedAssets, err := downloadAssets(ctx, opts.Prompt.Messages, download, opts.SupportedURLs)
	if err != nil {
		return nil, err
	}

	// Build approval ID to tool call ID mapping
	approvalIDToToolCallID := make(map[string]string)
	for _, msg := range opts.Prompt.Messages {
		if msg.Role == message.RoleAssistant && msg.Content.IsMultiPart() {
			for _, part := range msg.Content.Parts {
				if tar, ok := part.(message.ToolApprovalRequestPart); ok {
					approvalIDToToolCallID[tar.ApprovalID] = tar.ToolCallID
				}
			}
		}
	}

	// Build set of approved tool call IDs
	approvedToolCallIDs := make(map[string]bool)
	for _, msg := range opts.Prompt.Messages {
		if msg.Role == message.RoleTool {
			for _, part := range msg.Content.Parts {
				if tar, ok := part.(message.ToolApprovalResponsePart); ok {
					if toolCallID, exists := approvalIDToToolCallID[tar.ApprovalID]; exists {
						approvedToolCallIDs[toolCallID] = true
					}
				}
			}
		}
	}

	// Build the messages list
	var messages []LanguageModelMessage

	// Add system message(s) first
	if opts.Prompt.System != nil {
		switch sys := opts.Prompt.System.(type) {
		case string:
			messages = append(messages, LanguageModelMessage{
				Role:    "system",
				Content: sys,
			})
		case SystemMessage:
			messages = append(messages, LanguageModelMessage{
				Role:            "system",
				Content:         sys.Content,
				ProviderOptions: sys.ProviderOptions,
			})
		case []SystemMessage:
			for _, s := range sys {
				messages = append(messages, LanguageModelMessage{
					Role:            "system",
					Content:         s.Content,
					ProviderOptions: s.ProviderOptions,
				})
			}
		}
	}

	// Convert each message
	for _, msg := range opts.Prompt.Messages {
		converted := convertToLanguageModelMessage(msg, downloadedAssets)
		messages = append(messages, converted)
	}

	// Combine consecutive tool messages
	combinedMessages := combineToolMessages(messages)

	// Validate tool calls have matching results
	if err := validateToolResults(combinedMessages, approvedToolCallIDs); err != nil {
		return nil, err
	}

	// Filter out empty tool messages
	var result []LanguageModelMessage
	for _, msg := range combinedMessages {
		if msg.Role == "tool" {
			if parts, ok := msg.Content.([]LanguageModelPart); ok && len(parts) == 0 {
				continue
			}
		}
		result = append(result, msg)
	}

	return result, nil
}

// convertToLanguageModelMessage converts a single message to language model format.
func convertToLanguageModelMessage(msg message.Message, downloadedAssets map[string]*DownloadedAsset) LanguageModelMessage {
	switch msg.Role {
	case message.RoleSystem:
		return LanguageModelMessage{
			Role:    "system",
			Content: getTextFromContent(msg.Content),
		}

	case message.RoleUser:
		if !msg.Content.IsMultiPart() {
			return LanguageModelMessage{
				Role: "user",
				Content: []LanguageModelPart{{
					Type: "text",
					Text: msg.Content.Text,
				}},
			}
		}

		var parts []LanguageModelPart
		for _, part := range msg.Content.Parts {
			converted := convertUserPart(part, downloadedAssets)
			// Filter out empty text parts
			if converted.Type == "text" && converted.Text == "" {
				continue
			}
			parts = append(parts, converted)
		}
		return LanguageModelMessage{
			Role:    "user",
			Content: parts,
		}

	case message.RoleAssistant:
		if !msg.Content.IsMultiPart() {
			return LanguageModelMessage{
				Role: "assistant",
				Content: []LanguageModelPart{{
					Type: "text",
					Text: msg.Content.Text,
				}},
			}
		}

		var parts []LanguageModelPart
		for _, part := range msg.Content.Parts {
			converted := convertAssistantPart(part)
			if converted == nil {
				continue
			}
			// Filter out empty text parts (no text and no provider options)
			if converted.Type == "text" && converted.Text == "" && len(converted.ProviderOptions) == 0 {
				continue
			}
			parts = append(parts, *converted)
		}
		return LanguageModelMessage{
			Role:    "assistant",
			Content: parts,
		}

	case message.RoleTool:
		var parts []LanguageModelPart
		for _, part := range msg.Content.Parts {
			converted := convertToolPart(part)
			if converted == nil {
				continue
			}
			parts = append(parts, *converted)
		}
		return LanguageModelMessage{
			Role:    "tool",
			Content: parts,
		}

	default:
		return LanguageModelMessage{
			Role:    string(msg.Role),
			Content: getTextFromContent(msg.Content),
		}
	}
}

// convertUserPart converts a user message part.
func convertUserPart(part message.Part, downloadedAssets map[string]*DownloadedAsset) LanguageModelPart {
	switch p := part.(type) {
	case message.TextPart:
		return LanguageModelPart{
			Type: "text",
			Text: p.Text,
		}

	case message.ImagePart:
		data, mediaType := processDataContent(p.Image, p.MimeType, downloadedAssets)
		// Detect media type from data if not provided or is a wildcard
		if mediaType == "" || mediaType == "image/*" {
			if detected := detectImageMediaType(data); detected != "" {
				mediaType = detected
			} else if mediaType == "" {
				mediaType = "image/*"
			}
		}
		return LanguageModelPart{
			Type:      "file",
			Data:      data,
			MediaType: mediaType,
		}

	case message.FilePart:
		data, mediaType := processDataContent(p.Data, p.MimeType, downloadedAssets)
		if mediaType == "" && p.MimeType != "" {
			mediaType = p.MimeType
		}
		return LanguageModelPart{
			Type:      "file",
			Data:      data,
			MediaType: mediaType,
			Filename:  p.Filename,
		}

	default:
		return LanguageModelPart{Type: "unknown"}
	}
}

// convertAssistantPart converts an assistant message part.
func convertAssistantPart(part message.Part) *LanguageModelPart {
	switch p := part.(type) {
	case message.TextPart:
		return &LanguageModelPart{
			Type: "text",
			Text: p.Text,
		}

	case message.ToolCallPart:
		return &LanguageModelPart{
			Type:       "tool-call",
			ToolCallID: p.ID,
			ToolName:   p.Name,
			Input:      p.Input,
		}

	case message.ReasoningPart:
		return &LanguageModelPart{
			Type: "reasoning",
			Text: p.Text,
		}

	case message.FilePart:
		return &LanguageModelPart{
			Type:      "file",
			Data:      p.Data,
			MediaType: p.MimeType,
			Filename:  p.Filename,
		}

	case message.ToolResultPart:
		return &LanguageModelPart{
			Type:       "tool-result",
			ToolCallID: p.ToolCallID,
			ToolName:   p.ToolName,
			Output:     p.Result,
		}

	case message.ToolApprovalRequestPart:
		// Filter out tool-approval-request parts
		return nil

	default:
		return nil
	}
}

// convertToolPart converts a tool message part.
func convertToolPart(part message.Part) *LanguageModelPart {
	switch p := part.(type) {
	case message.ToolResultPart:
		return &LanguageModelPart{
			Type:       "tool-result",
			ToolCallID: p.ToolCallID,
			ToolName:   p.ToolName,
			Output:     mapToolResultOutput(p.Result),
		}

	case message.ToolApprovalResponsePart:
		// Only include if provider-executed
		if !p.ProviderExecuted {
			return nil
		}
		return &LanguageModelPart{
			Type:       "tool-approval-response",
			ApprovalID: p.ApprovalID,
			Approved:   p.Approved,
			Reason:     p.Reason,
		}

	default:
		return nil
	}
}

// processDataContent processes data content (URL, base64, bytes) and returns normalized form.
func processDataContent(data string, declaredMediaType string, downloadedAssets map[string]*DownloadedAsset) (any, string) {
	// Check if it's a data URL
	if strings.HasPrefix(data, "data:") {
		return parseDataURL(data)
	}

	// Check if it's a URL
	if strings.HasPrefix(data, "http://") || strings.HasPrefix(data, "https://") {
		// Check if we downloaded this URL
		if asset, ok := downloadedAssets[data]; ok {
			mediaType := declaredMediaType
			if mediaType == "" || mediaType == "application/octet-stream" {
				mediaType = asset.MediaType
			}
			return asset.Data, mediaType
		}
		// Return as URL if not downloaded
		u, err := url.Parse(data)
		if err == nil {
			return u, declaredMediaType
		}
	}

	// Assume base64-encoded data
	return data, declaredMediaType
}

// parseDataURL parses a data URL and returns the data and media type.
func parseDataURL(dataURL string) (string, string) {
	// Format: data:[<mediatype>][;base64],<data>
	if !strings.HasPrefix(dataURL, "data:") {
		return dataURL, ""
	}

	rest := strings.TrimPrefix(dataURL, "data:")
	commaIdx := strings.Index(rest, ",")
	if commaIdx == -1 {
		return dataURL, ""
	}

	header := rest[:commaIdx]
	data := rest[commaIdx+1:]

	// Parse header
	mediaType := ""
	isBase64 := false

	parts := strings.Split(header, ";")
	for i, part := range parts {
		if i == 0 && part != "" && part != "base64" {
			mediaType = part
		}
		if part == "base64" {
			isBase64 = true
		}
	}

	if isBase64 {
		// Return base64 string directly (providers can handle it)
		return data, mediaType
	}

	// URL-encoded data
	decoded, err := url.QueryUnescape(data)
	if err != nil {
		return data, mediaType
	}
	return base64.StdEncoding.EncodeToString([]byte(decoded)), mediaType
}

// combineToolMessages combines consecutive tool messages into a single message.
func combineToolMessages(messages []LanguageModelMessage) []LanguageModelMessage {
	var result []LanguageModelMessage

	for _, msg := range messages {
		if msg.Role != "tool" {
			result = append(result, msg)
			continue
		}

		// Check if last message is also a tool message
		if len(result) > 0 && result[len(result)-1].Role == "tool" {
			// Combine parts
			lastMsg := &result[len(result)-1]
			lastParts, _ := lastMsg.Content.([]LanguageModelPart)
			newParts, _ := msg.Content.([]LanguageModelPart)
			lastMsg.Content = append(lastParts, newParts...)
		} else {
			result = append(result, msg)
		}
	}

	return result
}

// validateToolResults validates that all tool calls have matching results.
func validateToolResults(messages []LanguageModelMessage, approvedToolCallIDs map[string]bool) error {
	toolCallIDs := make(map[string]bool)

	for _, msg := range messages {
		switch msg.Role {
		case "assistant":
			parts, ok := msg.Content.([]LanguageModelPart)
			if !ok {
				continue
			}
			for _, part := range parts {
				if part.Type == "tool-call" && !part.ProviderExecuted {
					toolCallIDs[part.ToolCallID] = true
				}
			}

		case "tool":
			parts, ok := msg.Content.([]LanguageModelPart)
			if !ok {
				continue
			}
			for _, part := range parts {
				if part.Type == "tool-result" {
					delete(toolCallIDs, part.ToolCallID)
				}
			}

		case "user", "system":
			// Before user/system messages, remove approved tool calls and check
			for id := range approvedToolCallIDs {
				delete(toolCallIDs, id)
			}
			if len(toolCallIDs) > 0 {
				return &errors.MissingToolResultsError{
					ToolCallIDs: mapKeys(toolCallIDs),
				}
			}
		}
	}

	// Final check
	for id := range approvedToolCallIDs {
		delete(toolCallIDs, id)
	}
	if len(toolCallIDs) > 0 {
		return &errors.MissingToolResultsError{
			ToolCallIDs: mapKeys(toolCallIDs),
		}
	}

	return nil
}

// downloadAssets downloads images/files from URLs that the model doesn't support.
func downloadAssets(ctx context.Context, messages []message.Message, download DownloadFunc, supportedURLs SupportedURLs) (map[string]*DownloadedAsset, error) {
	var plans []DownloadPlan

	for _, msg := range messages {
		if msg.Role != message.RoleUser || !msg.Content.IsMultiPart() {
			continue
		}

		for _, part := range msg.Content.Parts {
			var data string
			var mediaType string

			switch p := part.(type) {
			case message.ImagePart:
				data = p.Image
				mediaType = p.MimeType
				if mediaType == "" {
					mediaType = "image/*"
				}
			case message.FilePart:
				data = p.Data
				mediaType = p.MimeType
			default:
				continue
			}

			// Check if it's a URL
			if !strings.HasPrefix(data, "http://") && !strings.HasPrefix(data, "https://") {
				continue
			}

			u, err := url.Parse(data)
			if err != nil {
				continue
			}

			plans = append(plans, DownloadPlan{
				URL:                   u,
				IsURLSupportedByModel: isURLSupported(data, mediaType, supportedURLs),
			})
		}
	}

	if len(plans) == 0 {
		return make(map[string]*DownloadedAsset), nil
	}

	// Download in parallel
	results := download(ctx, plans)

	// Build result map
	assets := make(map[string]*DownloadedAsset)
	for i, result := range results {
		if result != nil {
			assets[plans[i].URL.String()] = result
		}
	}

	return assets, nil
}

// isURLSupported checks if a URL is supported by the model.
func isURLSupported(urlStr, mediaType string, supportedURLs SupportedURLs) bool {
	// Check specific media type patterns
	if patterns, ok := supportedURLs[mediaType]; ok {
		for _, pattern := range patterns {
			if pattern.MatchString(urlStr) {
				return true
			}
		}
	}

	// Check wildcard patterns (e.g., "image/*" matches "image/png")
	for key, patterns := range supportedURLs {
		if key == "*" || matchMediaType(key, mediaType) {
			for _, pattern := range patterns {
				if pattern.MatchString(urlStr) {
					return true
				}
			}
		}
	}

	return false
}

// matchMediaType checks if a pattern like "image/*" matches a media type like "image/png".
func matchMediaType(pattern, mediaType string) bool {
	if pattern == "*" || pattern == "*/*" {
		return true
	}

	patternParts := strings.Split(pattern, "/")
	mediaParts := strings.Split(mediaType, "/")

	if len(patternParts) != 2 || len(mediaParts) != 2 {
		return false
	}

	if patternParts[0] != mediaParts[0] && patternParts[0] != "*" {
		return false
	}

	if patternParts[1] != mediaParts[1] && patternParts[1] != "*" {
		return false
	}

	return true
}

// DefaultDownload is the default download function.
func DefaultDownload(ctx context.Context, plans []DownloadPlan) []*DownloadedAsset {
	results := make([]*DownloadedAsset, len(plans))

	for i, plan := range plans {
		// Skip if URL is already supported
		if plan.IsURLSupportedByModel {
			continue
		}

		// Download the asset
		asset, err := downloadURL(ctx, plan.URL)
		if err != nil {
			continue
		}
		results[i] = asset
	}

	return results
}

// downloadURL downloads content from a URL.
func downloadURL(ctx context.Context, u *url.URL) (*DownloadedAsset, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("download failed with status " + resp.Status)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	mediaType := resp.Header.Get("Content-Type")
	// Strip charset and other parameters
	if idx := strings.Index(mediaType, ";"); idx != -1 {
		mediaType = strings.TrimSpace(mediaType[:idx])
	}

	return &DownloadedAsset{
		Data:      data,
		MediaType: mediaType,
	}, nil
}

// detectImageMediaType detects image media type from binary data.
func detectImageMediaType(data any) string {
	var bytes []byte

	switch d := data.(type) {
	case []byte:
		bytes = d
	case string:
		// Try to decode base64
		decoded, err := base64.StdEncoding.DecodeString(d)
		if err != nil {
			return ""
		}
		bytes = decoded
	default:
		return ""
	}

	if len(bytes) < 3 {
		return ""
	}

	// JPEG: FF D8 FF
	if bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF {
		return "image/jpeg"
	}

	if len(bytes) < 4 {
		return ""
	}

	// PNG: 89 50 4E 47 0D 0A 1A 0A
	if bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4E && bytes[3] == 0x47 {
		return "image/png"
	}

	// GIF: 47 49 46 38
	if bytes[0] == 0x47 && bytes[1] == 0x49 && bytes[2] == 0x46 && bytes[3] == 0x38 {
		return "image/gif"
	}

	// WebP: 52 49 46 46 ... 57 45 42 50
	if len(bytes) > 11 &&
		bytes[0] == 0x52 && bytes[1] == 0x49 && bytes[2] == 0x46 && bytes[3] == 0x46 &&
		bytes[8] == 0x57 && bytes[9] == 0x45 && bytes[10] == 0x42 && bytes[11] == 0x50 {
		return "image/webp"
	}

	return ""
}

// mapToolResultOutput maps tool result output to the language model format.
func mapToolResultOutput(output any) any {
	// For now, pass through the output
	// ai-sdk has more complex mapping for content types
	return output
}

// getTextFromContent extracts text from message content.
func getTextFromContent(content message.Content) string {
	if content.Text != "" {
		return content.Text
	}
	var texts []string
	for _, part := range content.Parts {
		if tp, ok := part.(message.TextPart); ok {
			texts = append(texts, tp.Text)
		}
	}
	return strings.Join(texts, "")
}

// mapKeys returns the keys of a map as a slice.
func mapKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
