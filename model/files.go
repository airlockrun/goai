package model

import (
	"context"
	"io"
	"time"

	"github.com/airlockrun/goai/stream"
)

// Files uploads provider-managed files. Other operations are optional capabilities.
type Files interface {
	Provider() string
	UploadFile(context.Context, *UploadFileOptions) (*UploadFileResult, error)
}

type FileMetadataReader interface {
	GetFileMetadata(context.Context, *FileOptions) (*FileMetadata, error)
}

type FileDownloader interface {
	DownloadFile(context.Context, *FileOptions) (*DownloadFileResult, error)
}

type FileDeleter interface {
	DeleteFile(context.Context, *FileOptions) (*DeleteFileResult, error)
}

// ProviderReference can be used directly as message.FileDataReference.Reference.
type ProviderReference = map[string]any

type UploadFileOptions struct {
	// Data is consumed during upload. The caller owns and closes the reader.
	Data            io.Reader
	MediaType       string
	Filename        string
	Headers         map[string]string
	ProviderOptions map[string]any
}

type FileOptions struct {
	File            ProviderReference
	Headers         map[string]string
	ProviderOptions map[string]any
}

type FileMetadata struct {
	ProviderReference ProviderReference
	Filename          string
	MediaType         string
	ByteSize          *int64
	CreatedAt         *time.Time
	ExpiresAt         *time.Time
	ProviderMetadata  map[string]any
	Warnings          []stream.Warning
}

type UploadFileResult = FileMetadata

type DownloadFileResult struct {
	// Content must be closed by the caller, including on partial reads.
	Content   io.ReadCloser
	MediaType string
	Warnings  []stream.Warning
}

type DeleteFileResult struct {
	ProviderReference ProviderReference
	Deleted           bool
	Warnings          []stream.Warning
}
