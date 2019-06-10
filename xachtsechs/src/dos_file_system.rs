use crate::dos_error_codes::DosErrorCode;

use std::io::Read;

pub trait DosFileSystem : std::fmt::Debug {
	/// Returns a file handle if successful. Error code if not.
	fn create(&mut self, filename: Vec<u8>, attributes: u16) -> Result<u16, DosErrorCode>;
	/// Returns a file handle if successful. Error code if not.
	fn open(&mut self, filename: Vec<u8>, access_mode: DosFileAccessMode) -> Result<u16, DosErrorCode>;
	/// Retruns error code if close failed.
	fn close(&mut self, handle: u16) -> Result<(), DosErrorCode>;
	/// Returns the byte count read. Error code if read failed.
	fn read(&mut self, handle: u16, destination: &mut [u8]) -> Result<u16, DosErrorCode>;
	/// Returns the byte count written. Error code if write failed.
	fn write(&mut self, handle: u16, data: &[u8]) -> Result<u16, DosErrorCode>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DosFileAccessMode {
	ReadOnly,
	WriteOnly,
	ReadWrite,
}

#[derive(Debug)]
pub struct StandardDosFileSystem {
	root_path: std::path::PathBuf,
	file_handles: Vec<Option<std::fs::File>>,
}

impl StandardDosFileSystem {
	pub fn new(root_path: std::path::PathBuf) -> StandardDosFileSystem {
		StandardDosFileSystem {
			root_path,
			file_handles: vec![],
		}
	}
	
	fn get_empty_slot(&mut self) -> usize {
		match self.file_handles.iter().position(|ref slot| slot.is_none()) {
			Some(pos) => pos,
			None => {
				let pos = self.file_handles.len();
				self.file_handles.push(None);
				pos
			}
		}
	}
	
	fn get_real_filepath(&self, filename: &[u8]) -> std::path::PathBuf {
		if filename.contains(&b'\\') {
			unimplemented!("DOS directory mapping to real directories");
		}
		let mut string_filename = String::from_utf8_lossy(filename).into_owned();
		
		if let Ok(read_dir) = std::fs::read_dir(&self.root_path) {
			for dir_file in read_dir {
				if let Ok(dir_file_entry) = dir_file {
					if let Ok(dir_file_entry_name) = dir_file_entry.file_name().into_string() {
						if dir_file_entry_name.to_uppercase() == string_filename.to_uppercase() {
							string_filename = dir_file_entry_name;
						}
					}
				}
			}
		}
		self.root_path.join(string_filename)
	}
}

fn std_file_error_to_dos_error(err: std::io::Error) -> DosErrorCode {
	match err.kind() {
		std::io::ErrorKind::NotFound => DosErrorCode::FileNotFound,
		std::io::ErrorKind::PermissionDenied => DosErrorCode::AccessDenied,
		std::io::ErrorKind::AlreadyExists => DosErrorCode::FileAlreadyExists,
		_ => {
			eprintln!("Unexpected file error: {:?}", err);
			DosErrorCode::PathNotFound
		}
	}
}

impl DosFileSystem for StandardDosFileSystem {
	fn create(&mut self, filename: Vec<u8>, attributes: u16) -> Result<u16, DosErrorCode> {
		let real_filepath = self.get_real_filepath(&filename);
		let slot = self.get_empty_slot();
		match std::fs::File::create(real_filepath) {
			Ok(file) => {
				self.file_handles[slot] = Some(file);
				Ok(slot as u16 + 1)
			}
			Err(err) => Err(std_file_error_to_dos_error(err)),
		}
	}
	
	fn open(&mut self, filename: Vec<u8>, access_mode: DosFileAccessMode) -> Result<u16, DosErrorCode> {
		// TODO: 776655
		let real_filepath = self.get_real_filepath(&filename);
		let slot = self.get_empty_slot();
		
		let mut open_options = std::fs::OpenOptions::new();

		open_options
			.read(access_mode == DosFileAccessMode::ReadOnly || access_mode == DosFileAccessMode::ReadWrite)
			.write(access_mode == DosFileAccessMode::WriteOnly || access_mode == DosFileAccessMode::ReadWrite)
			.create(access_mode == DosFileAccessMode::WriteOnly || access_mode == DosFileAccessMode::ReadWrite);
		
		match open_options.open(real_filepath) {
			Ok(file) => {
				self.file_handles[slot] = Some(file);
				Ok(slot as u16 + 1)
			}
			Err(err) => Err(std_file_error_to_dos_error(err)),
		}
	}
	
	fn close(&mut self, handle: u16) -> Result<(), DosErrorCode> {
		unimplemented!()
	}
	
	fn read(&mut self, handle: u16, destination: &mut [u8]) -> Result<u16, DosErrorCode> {
		if handle == 0 {
			Err(DosErrorCode::InvalidFileHandle)
		} else {
			let handle_index = (handle - 1) as usize;
			if let Some(Some(ref mut file)) = self.file_handles.get_mut(handle_index) {
				match file.read(destination) {
					Ok(read_count) => Ok(read_count as u16),
					Err(err) => Err(std_file_error_to_dos_error(err)),
				}
			} else {
				Err(DosErrorCode::InvalidFileHandle)
			}
		}
	}
	
	fn write(&mut self, handle: u16, data: &[u8]) -> Result<u16, DosErrorCode> {
		unimplemented!()
	}
}
