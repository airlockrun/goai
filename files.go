package goai

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	goaierrors "github.com/airlockrun/goai/errors"
	"github.com/airlockrun/goai/model"
)

type UploadFileInput struct {
	Files model.Files
	// Supply exactly one of Data, DataReader, or DataBase64.
	Data       []byte
	DataReader io.Reader
	DataBase64 string
	// MediaType is detected for inline data when omitted. Readers default to application/octet-stream.
	MediaType       string
	Filename        string
	Headers         map[string]string
	ProviderOptions map[string]any
}

type FileInput struct {
	Files           model.Files
	File            model.ProviderReference
	Headers         map[string]string
	ProviderOptions map[string]any
}

type UploadFileResult = model.UploadFileResult
type FileMetadata = model.FileMetadata
type DownloadFileResult = model.DownloadFileResult
type DeleteFileResult = model.DeleteFileResult

func UploadFile(ctx context.Context, input UploadFileInput) (*UploadFileResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if input.Files == nil {
		return nil, errors.New("files capability is required")
	}
	count := 0
	var data io.Reader
	var inline []byte
	if input.Data != nil {
		count++
		inline = input.Data
		data = bytes.NewReader(input.Data)
	}
	if input.DataReader != nil {
		count++
		data = input.DataReader
	}
	if input.DataBase64 != "" {
		count++
		decoded, err := base64.StdEncoding.DecodeString(input.DataBase64)
		if err != nil {
			return nil, fmt.Errorf("file base64: %w", err)
		}
		data = bytes.NewReader(decoded)
		inline = decoded
	}
	if count != 1 {
		return nil, errors.New("exactly one file data source is required")
	}
	if input.MediaType == "" {
		input.MediaType = "application/octet-stream"
		if len(inline) > 0 {
			input.MediaType = strings.Split(http.DetectContentType(inline), ";")[0]
		}
	}
	return input.Files.UploadFile(ctx, &model.UploadFileOptions{Data: data, MediaType: input.MediaType, Filename: input.Filename, Headers: input.Headers, ProviderOptions: input.ProviderOptions})
}

func GetFileMetadata(ctx context.Context, input FileInput) (*FileMetadata, error) {
	if input.Files == nil {
		return nil, errors.New("files capability is required")
	}
	files, ok := input.Files.(model.FileMetadataReader)
	if !ok {
		return nil, fmt.Errorf("%w: %s file metadata", goaierrors.ErrUnsupported, input.Files.Provider())
	}
	return files.GetFileMetadata(ctx, &model.FileOptions{File: input.File, Headers: input.Headers, ProviderOptions: input.ProviderOptions})
}

func DownloadFile(ctx context.Context, input FileInput) (*DownloadFileResult, error) {
	if input.Files == nil {
		return nil, errors.New("files capability is required")
	}
	files, ok := input.Files.(model.FileDownloader)
	if !ok {
		return nil, fmt.Errorf("%w: %s file download", goaierrors.ErrUnsupported, input.Files.Provider())
	}
	return files.DownloadFile(ctx, &model.FileOptions{File: input.File, Headers: input.Headers, ProviderOptions: input.ProviderOptions})
}

func DeleteFile(ctx context.Context, input FileInput) (*DeleteFileResult, error) {
	if input.Files == nil {
		return nil, errors.New("files capability is required")
	}
	files, ok := input.Files.(model.FileDeleter)
	if !ok {
		return nil, fmt.Errorf("%w: %s file deletion", goaierrors.ErrUnsupported, input.Files.Provider())
	}
	return files.DeleteFile(ctx, &model.FileOptions{File: input.File, Headers: input.Headers, ProviderOptions: input.ProviderOptions})
}
