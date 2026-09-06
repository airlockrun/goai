package openai

import (
	"strconv"

	"github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/model"
)

type FilesOptions struct {
	Purpose      string `json:"purpose"`
	ExpiresAfter *int64 `json:"expiresAfter"`
}

// Files returns OpenAI file upload, metadata, download, and deletion capabilities.
func (p *Provider) Files() model.Files {
	headers := map[string]string{"Authorization": "Bearer " + p.opts.APIKey}
	if p.opts.Organization != "" {
		headers["OpenAI-Organization"] = p.opts.Organization
	}
	if p.opts.Project != "" {
		headers["OpenAI-Project"] = p.opts.Project
	}
	headers = internal.FilesHeaders(headers, p.opts.Headers)
	return &internal.FilesClient{ProviderID: "openai", BaseURL: p.opts.BaseURL, Headers: headers, UploadFields: func(options map[string]any) ([][2]string, error) {
		opts, err := internal.ParseFilesOptions[FilesOptions](options, "openai")
		if err != nil {
			return nil, err
		}
		purpose := opts.Purpose
		if purpose == "" {
			purpose = "assistants"
		}
		fields := [][2]string{{"purpose", purpose}}
		if opts.ExpiresAfter != nil {
			fields = append(fields, [2]string{"expires_after[anchor]", "created_at"}, [2]string{"expires_after[seconds]", strconv.FormatInt(*opts.ExpiresAfter, 10)})
		}
		return fields, nil
	}}
}
