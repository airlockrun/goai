package google

import (
	"encoding/json"
	"strings"

	"github.com/airlockrun/goai/tool"
)

// prepareGeminiTools splits a tool list into function declarations and
// provider-defined tools, emitting Gemini-compatible tool entries for each.
//
// Mirrors ai-sdk's google-prepare-tools.ts. Model-version-specific behavior:
//   - Gemini 2+: googleSearch / googleMaps / enterpriseWebSearch / urlContext /
//     codeExecution emit bare {...: {}} entries.
//   - gemini-1.5-flash (not -8b): googleSearch emits googleSearchRetrieval with
//     an optional dynamicRetrievalConfig from the tool's Args.
//   - Older models: googleSearch emits an empty googleSearchRetrieval; other
//     provider tools are silently dropped (no warning plumbing in goai yet).
func prepareGeminiTools(tools []tool.Tool, modelID string) []geminiTool {
	if len(tools) == 0 {
		return nil
	}

	isGemini2OrNewer := strings.Contains(modelID, "gemini-2") ||
		strings.Contains(modelID, "gemini-3") ||
		strings.Contains(modelID, "gemini-flash-latest") ||
		strings.Contains(modelID, "gemini-flash-lite-latest") ||
		strings.Contains(modelID, "gemini-pro-latest")
	supportsDynamicRetrieval := strings.Contains(modelID, "gemini-1.5-flash") &&
		!strings.Contains(modelID, "-8b")

	var providerTools []geminiTool
	var functionTools []tool.Tool

	for _, t := range tools {
		if t.Type != "provider" {
			functionTools = append(functionTools, t)
			continue
		}
		switch t.ProviderID {
		case ToolIDGoogleSearch:
			switch {
			case isGemini2OrNewer:
				gs := &geminiGoogleSearch{}
				if len(t.Args) > 0 {
					var opts GoogleSearchOptions
					if err := json.Unmarshal(t.Args, &opts); err == nil {
						if opts.SearchTypes != nil {
							gs.SearchTypes = &geminiGoogleSearchTypes{
								WebSearch:   opts.SearchTypes.WebSearch,
								ImageSearch: opts.SearchTypes.ImageSearch,
							}
						}
						if opts.TimeRangeFilter != nil {
							gs.TimeRangeFilter = &geminiGoogleSearchTimeRange{
								StartTime: opts.TimeRangeFilter.StartTime,
								EndTime:   opts.TimeRangeFilter.EndTime,
							}
						}
					}
				}
				providerTools = append(providerTools, geminiTool{GoogleSearch: gs})
			case supportsDynamicRetrieval:
				cfg := &geminiGoogleSearchRetrieval{}
				if len(t.Args) > 0 {
					var opts GoogleSearchOptions
					if err := json.Unmarshal(t.Args, &opts); err == nil {
						cfg.DynamicRetrievalConfig = &geminiDynamicRetrievalConfig{
							Mode:             opts.Mode,
							DynamicThreshold: opts.DynamicThreshold,
						}
					}
				}
				providerTools = append(providerTools, geminiTool{GoogleSearchRetrieval: cfg})
			default:
				providerTools = append(providerTools, geminiTool{GoogleSearchRetrieval: &geminiGoogleSearchRetrieval{}})
			}
		case ToolIDGoogleMaps:
			if isGemini2OrNewer {
				providerTools = append(providerTools, geminiTool{GoogleMaps: &struct{}{}})
			}
		case ToolIDEnterpriseWebSearch:
			if isGemini2OrNewer {
				providerTools = append(providerTools, geminiTool{EnterpriseWebSearch: &struct{}{}})
			}
		case ToolIDURLContext:
			if isGemini2OrNewer {
				providerTools = append(providerTools, geminiTool{URLContext: &struct{}{}})
			}
		case ToolIDCodeExecution:
			if isGemini2OrNewer {
				providerTools = append(providerTools, geminiTool{CodeExecution: &struct{}{}})
			}
		}
	}

	// Function declarations go into a single entry, matching ai-sdk's shape.
	// If both function and provider tools are present we emit both — Gemini
	// warns on mixed use but accepts it; keeping goai's default permissive.
	if len(functionTools) > 0 {
		providerTools = append(providerTools, geminiTool{
			FunctionDeclarations: convertToGeminiFunctions(functionTools),
		})
	}
	return providerTools
}
