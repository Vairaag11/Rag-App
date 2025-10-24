# Documents Folder

This folder supports multiple document formats for the knowledge base:

## Supported File Types:
- **`.txt`** - Plain text files
- **`.md`** - Markdown files  
- **`.pdf`** - PDF documents (requires `pypdf`)
- **`.docx`** - Word documents (requires `docx2txt`)

## Usage:
1. Place your knowledge documents in this folder
2. Delete the `vector_store/` folder to force rebuild
3. Restart the application

## File Processing:
- Each file is loaded with appropriate loader
- Source filename is preserved in metadata
- Files are automatically chunked and embedded
- Unsupported files are skipped with warnings

## Installation for Additional Formats:
```bash
# For PDF support
pip install pypdf

# For Word document support  
pip install docx2txt
```

## Fallback:
If this folder is empty or doesn't exist, the system falls back to the single `knowledge_base_clickatell.txt` file.