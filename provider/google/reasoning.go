package google

import (
	"encoding/json"
	"fmt"
	"math"
	"path"
	"regexp"
	"strconv"
	"strings"
)

var olderGemini = regexp.MustCompile(`^gemini-[12]([.-]|$)|^gemini-pro(-vision)?$|^gemini-robotics-er-1\.5([.-]|$)`)
var flashVersion = regexp.MustCompile(`^gemini-(\d+)\.(\d+)-flash($|-)`)

// ThinkingConfiguration resolves uniform effort and explicit Gemini thinking options.
func ThinkingConfiguration(modelID, effort string, explicit any) (map[string]any, error) {
	config := map[string]any{}
	name := strings.ToLower(path.Base(modelID))
	if effort != "" && effort != "provider-default" {
		ratios := map[string]float64{"none": 0, "minimal": 0.02, "low": 0.1, "medium": 0.3, "high": 0.6, "xhigh": 0.9}
		ratio, ok := ratios[effort]
		if !ok {
			return nil, fmt.Errorf("unsupported reasoning effort %q", effort)
		}
		if strings.HasPrefix(name, "gemini-") && !olderGemini.MatchString(name) && !strings.Contains(name, "gemini-3-pro-image") {
			level := effort
			if effort == "none" || effort == "minimal" {
				level = "minimal"
				match := flashVersion.FindStringSubmatch(name)
				if name == "gemini-flash-latest" {
					level = "low"
				}
				if len(match) > 0 && !strings.Contains(name, "-flash-lite") {
					major, _ := strconv.Atoi(match[1])
					minor, _ := strconv.Atoi(match[2])
					if major > 3 || major == 3 && minor >= 7 {
						level = "low"
					}
				}
			}
			if level == "xhigh" {
				level = "high"
			}
			config["thinkingLevel"] = level
		} else {
			max := 24576
			if strings.Contains(name, "2.5-pro") || strings.Contains(name, "gemini-3-pro-image") {
				max = 32768
			}
			config["thinkingBudget"] = min(max, int(math.Round(65536*ratio)))
		}
	}
	if explicit != nil {
		data, err := json.Marshal(explicit)
		if err != nil {
			return nil, err
		}
		var values map[string]any
		if err := json.Unmarshal(data, &values); err != nil {
			return nil, err
		}
		if _, ok := values["thinkingLevel"]; ok {
			delete(config, "thinkingBudget")
		}
		if _, ok := values["thinkingBudget"]; ok {
			delete(config, "thinkingLevel")
		}
		for k, v := range values {
			config[k] = v
		}
	}
	if len(config) == 0 {
		return nil, nil
	}
	return config, nil
}
