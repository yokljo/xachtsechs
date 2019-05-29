pub trait DosFileSystem : std::fmt::Debug {
	/// Returns a file handle if successful. Error code if not.
	fn create(&mut self, filename: Vec<u8>, attributes: u16) -> Result<u16, u16>;
	/// Returns a file handle if successful. Error code if not.
	fn open(&mut self, filename: Vec<u8>, access_modes: u8) -> Result<u16, u16>;
	/// Retruns error code if close failed.
	fn close(&mut self, handle: u16) -> Result<(), u16>;
	/// Returns the byte count read. Error code if read failed.
	fn read(&mut self, handle: u16, count: u16, destination: &[u8]) -> Result<u16, u16>;
	/// Returns the byte count written. Error code if write failed.
	fn write(&mut self, handle: u16, count: u16, data: &[u8]) -> Result<u16, u16>;
}

#[derive(Debug)]
pub struct StandardDosFileSystem {
	file_handles: Vec<Option<std::fs::File>>,
}

impl StandardDosFileSystem {
	pub fn new() -> StandardDosFileSystem {
		StandardDosFileSystem {
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
	
	fn get_real_filename(&self, filename: &[u8]) -> String {
		"./dos/".to_string() + &String::from_utf8_lossy(filename)
	}
}

impl DosFileSystem for StandardDosFileSystem {
	fn create(&mut self, filename: Vec<u8>, attributes: u16) -> Result<u16, u16> {
		let string_filename = self.get_real_filename(&filename);
		let slot = self.get_empty_slot();
		match std::fs::File::create(string_filename) {
			Ok(file) => {
				self.file_handles[slot] = Some(file);
				Ok(slot as u16)
			}
			Err(_) => Err(0x03),
		}
	}
	fn open(&mut self, filename: Vec<u8>, access_modes: u8) -> Result<u16, u16> {
		let string_filename = self.get_real_filename(&filename);
		let slot = self.get_empty_slot();
		match std::fs::File::open(string_filename) {
			Ok(file) => {
				self.file_handles[slot] = Some(file);
				Ok(slot as u16)
			}
			Err(_) => Err(0x03),
		}
	}
	fn close(&mut self, handle: u16) -> Result<(), u16> {
		unimplemented!()
	}
	fn read(&mut self, handle: u16, count: u16, destination: &[u8]) -> Result<u16, u16> {
		unimplemented!()
	}
	fn write(&mut self, handle: u16, count: u16, data: &[u8]) -> Result<u16, u16> {
		unimplemented!()
	}
}
