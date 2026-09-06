package xai

import (
	"errors"
	"strconv"

	"github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/model"
)

type FilesOptions struct {
	TeamID       string `json:"teamId"`
	FilePath     string `json:"filePath"`
	ExpiresAfter *int64 `json:"expiresAfter"`
}

func (p *Provider) Files() model.Files {
	headers := map[string]string{"Authorization": "Bearer " + p.apiKey}
	headers = internal.FilesHeaders(headers, p.headers)
	return &internal.FilesClient{ProviderID: "xai", BaseURL: p.baseURL, Headers: headers, UploadFields: func(options map[string]any) ([][2]string, error) {
		opts, err := internal.ParseFilesOptions[FilesOptions](options, "xai")
		if err != nil {
			return nil, err
		}
		var fields [][2]string
		if opts.ExpiresAfter != nil {
			if *opts.ExpiresAfter < 3600 || *opts.ExpiresAfter > 2592000 {
				return nil, errors.New("xai expiresAfter must be between 3600 and 2592000 seconds")
			}
			fields = append(fields, [2]string{"expires_after", strconv.FormatInt(*opts.ExpiresAfter, 10)})
		}
		if opts.TeamID != "" {
			fields = append(fields, [2]string{"team_id", opts.TeamID})
		}
		return fields, nil
	}}
}
