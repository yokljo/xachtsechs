use crate::types::{DataLocation8, DataLocation16, Reg};
use crate::machine8086::Machine8086;
use crate::bios_loader::initialise_bios_data_area;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Seek;

// https://wiki.osdev.org/MZ

const EXE_PARAGRAPH_BYTES: usize = 16;
// The Program Segment Prefix is 256 bytes in size, which is 16 paragraphs.
const EXE_PROGRAM_SEGMENT_PREFIX_PARAGRAPHS: usize = 16;
const EXE_BLOCK_BYTES: usize = 512;
// This is the paragraph where the EXE file puts the code data.
const EXE_ORIGIN_PARAGRAPH: usize = 0x100;

#[derive(Debug)]
pub struct MzHeader {
	signature: u16,
	last_block_bytes: u16,
	file_block_count: u16,
	relocation_items: u16,
	header_paragraph_count: u16,
	minimum_memory_paragraphs: u16,
	maximum_memory_paragraphs: u16,
	initial_ss: u16,
	initial_sp: u16,
	checksum: u16,
	initial_ip: u16,
	initial_cs: u16,
	relocation_table: u16,
	overlay: u16,
	overlay_information: u16,
}

impl MzHeader {
	pub fn byte_size() -> usize {
		28
	}

	pub fn parse(stream: &mut std::io::Read) -> Result<MzHeader, String> {
		let signature = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read signature: {}", e))?;
		let last_block_bytes = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read last_block_bytes: {}", e))?;
		let file_block_count = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read file_block_count: {}", e))?;
		let relocation_items = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read relocation_items: {}", e))?;
		let header_paragraph_count = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read header_paragraph_count: {}", e))?;
		let minimum_memory_paragraphs = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read minimum_memory_paragraphs: {}", e))?;
		let maximum_memory_paragraphs = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read maximum_memory_paragraphs: {}", e))?;
		let initial_ss = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read initial_ss: {}", e))?;
		let initial_sp = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read initial_sp: {}", e))?;
		let checksum = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read checksum: {}", e))?;
		let initial_ip = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read initial_ip: {}", e))?;
		let initial_cs = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read initial_cs: {}", e))?;
		let relocation_table = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read relocation_table: {}", e))?;
		let overlay = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read overlay: {}", e))?;
		let overlay_information = stream.read_u16::<LittleEndian>().map_err(|e| format!("Failed to read overlay_information: {}", e))?;
		
		Ok(MzHeader {
			signature,
			last_block_bytes,
			file_block_count,
			relocation_items,
			header_paragraph_count,
			minimum_memory_paragraphs,
			maximum_memory_paragraphs,
			initial_ss,
			initial_sp,
			checksum,
			initial_ip,
			initial_cs,
			relocation_table,
			overlay,
			overlay_information,
		})
	}
	
	pub fn data_start(&self) -> usize {
		self.header_paragraph_count as usize * EXE_PARAGRAPH_BYTES
	}
	
	pub fn data_end(&self) -> usize {
		let subtract_bytes = if self.last_block_bytes > 0 {
			EXE_BLOCK_BYTES - self.last_block_bytes as usize
		} else {
			0
		};
		(self.file_block_count as usize * EXE_BLOCK_BYTES) - subtract_bytes
	}
	
	pub fn extract_data<StreamType>(&self, stream: &mut StreamType) -> Result<Vec<u8>, std::io::Error>
		where StreamType: std::io::Read + std::io::Seek
	{
		stream.seek(std::io::SeekFrom::Start(self.data_start() as u64));
		let data_length = self.data_end() - self.data_start();
		let mut result = vec![];
		result.resize(data_length, 0);
		stream.read(&mut result)?;
		Ok(result)
	}
	
	pub fn load_into_machine<StreamType>(&self, machine: &mut Machine8086, stream: &mut StreamType)
		where StreamType: std::io::Read + std::io::Seek
	{
		machine.set_reg_u16(Reg::SP, self.initial_sp);
		machine.set_reg_u16(Reg::IP, self.initial_ip);
		
		let segment_offset = (EXE_ORIGIN_PARAGRAPH + EXE_PROGRAM_SEGMENT_PREFIX_PARAGRAPHS) as u16;
		machine.set_reg_u16(Reg::SS, self.initial_ss + segment_offset);
		machine.set_reg_u16(Reg::CS, self.initial_cs + segment_offset);
		
		machine.set_reg_u16(Reg::DS, EXE_ORIGIN_PARAGRAPH as u16);
		machine.set_reg_u16(Reg::ES, EXE_ORIGIN_PARAGRAPH as u16);
		
		let exe_data = self.extract_data(stream).unwrap();
		machine.insert_contiguous_bytes(&exe_data, (EXE_ORIGIN_PARAGRAPH + 16) * EXE_PARAGRAPH_BYTES);
		
		initialise_bios_data_area(machine);
		initialise_dos_program_segment_prefix(machine, exe_data.len(), b"");
		
		/*for (i, b) in machine.memory[10000..20000].iter().enumerate() {
			println!("{}: {:02x}", i + 10000, b);
		}
		panic!();*/
	}
}

// https://en.wikipedia.org/wiki/Program_Segment_Prefix
fn initialise_dos_program_segment_prefix(machine: &mut Machine8086, program_size: usize, command_line_tail: &[u8]) -> Result<(), String> {
	// The DS register will be the PSP location when a program starts.
	let psp_start = (EXE_ORIGIN_PARAGRAPH * EXE_PARAGRAPH_BYTES) as u32; //machine.get_seg_origin(Reg::DS);
	// CP/M exit: Always 20h
	//machine.poke_u16(psp_start + 0x00, 0x20);
	// These values are probably all wrong:
	
	// Segment after the memeory allocated to the program.
	dbg!((psp_start, program_size));
	machine.poke_u16(psp_start + 0x02, 0xa000);
	
	// +1 for the 0x0d teminator character.
	let command_line_tail_len = command_line_tail.len() + 1;
	if command_line_tail_len > 0xff {
		return Err(format!("Command line tail too long: {}", command_line_tail.len()));
	}
	machine.poke_u8(psp_start + 0x80, command_line_tail_len as u8);
	let mut current_command_line_pos = psp_start + 0x81;
	for byte in command_line_tail {
		machine.poke_u8(current_command_line_pos, *byte);
		current_command_line_pos += 1;
	}
	machine.poke_u8(current_command_line_pos, 0x0d);
	
	Ok(())
}
