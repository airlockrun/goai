package goai_test

import (
	"bufio"
	"os"
	"strings"
)

func init() {
	// Load .env file if it exists
	loadEnvFile(".env")
	loadEnvFile("../.env") // Also check parent directory
}

func loadEnvFile(path string) {
	file, err := os.Open(path)
	if err != nil {
		return // File doesn't exist, skip
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Parse KEY=VALUE
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		// Remove surrounding quotes if present
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') ||
				(value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}

		// Only set if not already set (don't override existing env vars)
		if os.Getenv(key) == "" {
			os.Setenv(key, value)
		}
	}
}
