package anthropic

import (
	"context"

	"github.com/airlockrun/goai/internal"
	"github.com/airlockrun/goai/model"
)

type files struct{ client *internal.FilesClient }

func (p *Provider) Files() model.Files {
	headers := map[string]string{"anthropic-version": apiVersion, "anthropic-beta": "files-api-2025-04-14"}
	if p.opts.AuthScheme == "bearer" {
		headers["Authorization"] = "Bearer " + p.opts.APIKey
	} else {
		headers["x-api-key"] = p.opts.APIKey
	}
	headers = internal.FilesHeaders(headers, p.opts.Headers)
	return &files{client: &internal.FilesClient{ProviderID: "anthropic", BaseURL: p.opts.BaseURL, Headers: headers}}
}

func (f *files) Provider() string { return "anthropic" }
func (f *files) UploadFile(ctx context.Context, opts *model.UploadFileOptions) (*model.UploadFileResult, error) {
	return f.client.UploadFile(ctx, opts)
}
