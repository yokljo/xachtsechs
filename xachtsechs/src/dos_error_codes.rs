#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum DosErrorCode {
	FileNotFound = 0x02,
	PathNotFound = 0x03,
	NoFileHandlesLeft = 0x04,
	AccessDenied = 0x05,
	InvalidFileHandle = 0x06,
	InsufficientMemory = 0x08,
	InvalidFileAccessMode = 0x0c,
	FileAlreadyExists = 0x50,
}
