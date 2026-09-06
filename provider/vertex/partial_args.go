package vertex

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type partialArg struct {
	JSONPath     string   `json:"jsonPath"`
	StringValue  *string  `json:"stringValue"`
	NumberValue  *float64 `json:"numberValue"`
	BoolValue    *bool    `json:"boolValue"`
	WillContinue bool     `json:"willContinue"`
}

// setPartialArg builds nested arguments and appends continued string values.
func setPartialArg(node any, path string, value any) (any, error) {
	if path == "" {
		if text, ok := value.(string); ok {
			if previous, ok := node.(string); ok {
				return previous + text, nil
			}
		}
		return value, nil
	}
	if path[0] == '[' {
		end := strings.IndexByte(path, ']')
		if end < 0 {
			return nil, fmt.Errorf("invalid partial argument path %q", path)
		}
		index, err := strconv.Atoi(path[1:end])
		if err != nil || index < 0 || index > 100000 {
			return nil, fmt.Errorf("invalid partial argument index %q", path)
		}
		array, ok := node.([]any)
		if node != nil && !ok {
			return nil, errors.New("partial argument path conflicts with value")
		}
		for len(array) <= index {
			array = append(array, nil)
		}
		array[index], err = setPartialArg(array[index], path[end+1:], value)
		return array, err
	}
	path = strings.TrimPrefix(path, ".")
	end := strings.IndexAny(path, ".[")
	if end < 0 {
		end = len(path)
	}
	if end == 0 {
		return nil, fmt.Errorf("invalid partial argument path %q", path)
	}
	object, ok := node.(map[string]any)
	if node != nil && !ok {
		return nil, errors.New("partial argument path conflicts with value")
	}
	if object == nil {
		object = map[string]any{}
	}
	key := path[:end]
	child, err := setPartialArg(object[key], path[end:], value)
	if err != nil {
		return nil, err
	}
	object[key] = child
	return object, nil
}
