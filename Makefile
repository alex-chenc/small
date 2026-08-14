BINARY := small
GO ?= go

.PHONY: all
all:
	CGO_ENABLED=0 $(GO) build -trimpath -ldflags="-s -w" -o $(BINARY) .
