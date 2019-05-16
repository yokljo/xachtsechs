use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Seek;
use std::cmp::Ordering;
use num::FromPrimitive;
use num_derive::FromPrimitive;

// https://wiki.osdev.org/MZ

const EXE_PARAGRAPH_BYTES: usize = 16;
const EXE_BLOCK_BYTES: usize = 512;
// This is the paragraph where the EXE file puts the code data.
const EXE_ORIGIN_PARAGRAPH: usize = 0x100;

#[derive(Debug)]
struct MzHeader {
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
		
		let segment_offset = (EXE_ORIGIN_PARAGRAPH + EXE_PARAGRAPH_BYTES) as u16;
		machine.set_reg_u16(Reg::SS, self.initial_ss + segment_offset);
		machine.set_reg_u16(Reg::CS, self.initial_cs + segment_offset);
		
		machine.set_reg_u16(Reg::DS, EXE_ORIGIN_PARAGRAPH as u16);
		machine.set_reg_u16(Reg::ES, EXE_ORIGIN_PARAGRAPH as u16);
		
		let exe_data = self.extract_data(stream).unwrap();
		machine.insert_contiguous_bytes(&exe_data, (EXE_ORIGIN_PARAGRAPH + 16) * EXE_PARAGRAPH_BYTES);
	}
}

trait AnyUnsignedInt8086
	: std::ops::Add
	+ std::ops::AddAssign
	+ std::ops::Sub
	+ std::ops::SubAssign
	+ std::ops::BitOr<Self, Output = Self>
	+ std::ops::BitOrAssign
	+ std::ops::BitAnd<Self, Output = Self>
	+ std::ops::BitAndAssign
	+ std::ops::BitXor<Self, Output = Self>
	+ std::ops::BitXorAssign
	+ std::ops::Shl<u8, Output = Self>
	+ std::ops::Shr<u8, Output = Self>
	+ std::cmp::PartialOrd
	+ std::cmp::PartialEq
	+ std::fmt::Debug
	+ std::marker::Copy
	+ std::marker::Sized
{
	const BIT_SIZE: u8;
	const ZERO: Self;
	const ONE: Self;
	fn wrapping_add(self, rhs: Self) -> Self;
	fn wrapping_sub(self, rhs: Self) -> Self;
	fn rotate_left(self, n: u32) -> Self;
	fn rotate_right(self, n: u32) -> Self;
	fn most_significant_bit(self, dist_from_left: u8) -> bool;
	fn least_significant_bit(self, dist_from_right: u8) -> bool;
	fn least_significant_byte(self) -> u8;
}

impl AnyUnsignedInt8086 for u8 {
	const BIT_SIZE: u8 = 8;
	const ZERO: Self = 0;
	const ONE: Self = 1;
	fn wrapping_add(self, rhs: u8) -> u8 { u8::wrapping_add(self, rhs) }
	fn wrapping_sub(self, rhs: u8) -> u8 { u8::wrapping_sub(self, rhs) }
	fn rotate_left(self, n: u32) -> Self { u8::rotate_left(self, n) }
	fn rotate_right(self, n: u32) -> Self { u8::rotate_right(self, n) }
	fn most_significant_bit(self, dist_from_left: u8) -> bool {
		((self >> (7 - dist_from_left)) & 1) == 1
	}
	fn least_significant_bit(self, dist_from_right: u8) -> bool {
		((self >> dist_from_right) & 1) == 1
	}
	fn least_significant_byte(self) -> u8 { self }
}

impl AnyUnsignedInt8086 for u16 {
	const BIT_SIZE: u8 = 16;
	const ZERO: Self = 0;
	const ONE: Self = 1;
	fn wrapping_add(self, rhs: u16) -> u16 { u16::wrapping_add(self, rhs) }
	fn wrapping_sub(self, rhs: u16) -> u16 { u16::wrapping_sub(self, rhs) }
	fn rotate_left(self, n: u32) -> Self { u16::rotate_left(self, n) }
	fn rotate_right(self, n: u32) -> Self { u16::rotate_right(self, n) }
	fn most_significant_bit(self, dist_from_left: u8) -> bool {
		((self >> (15 - dist_from_left)) & 1) == 1
	}
	fn least_significant_bit(self, dist_from_right: u8) -> bool {
		((self >> dist_from_right) & 1) == 1
	}
	fn least_significant_byte(self) -> u8 { (self & 0xff) as u8 }
}

// https://stackoverflow.com/questions/9130349/how-many-registers-are-there-in-8086-8088
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
enum Reg {
	// Accumulator
	AX,
	// Base address
	BX,
	// Count
	CX,
	// Data
	DX,
	// Stack pointer
	SP,
	// (stack) Base pointer
	BP,
	// Source index
	SI,
	// Destination index
	DI,
	// Code segment
	CS,
	// Data segment
	DS,
	// Extra segment
	ES,
	// Stack segment
	SS,
	// Instruction pointer
	IP,
	// Comparison flags
	Flags,
}

const REG_COUNT: usize = 14;

#[derive(Debug, Copy, Clone, PartialEq)]
enum RegHalf {
	Low,
	High,
}

const REG8_MAPPING: [(Reg, RegHalf); 8] = [
	(Reg::AX, RegHalf::Low), (Reg::CX, RegHalf::Low), (Reg::DX, RegHalf::Low), (Reg::BX, RegHalf::Low),
	(Reg::AX, RegHalf::High), (Reg::CX, RegHalf::High), (Reg::DX, RegHalf::High), (Reg::BX, RegHalf::High),
];
const REG16_MAPPING: [Reg; 8] = [Reg::AX, Reg::CX, Reg::DX, Reg::BX, Reg::SP, Reg::BP, Reg::SI, Reg::DI];
const SREG_MAPPING: [Reg; 4] = [Reg::ES, Reg::CS, Reg::SS, Reg::DS];

impl Reg {
	fn reg16(index: usize) -> Option<Reg> {
		REG16_MAPPING.get(index).map(|v| *v)
	}
	
	fn reg8(index: usize) -> Option<(Reg, RegHalf)> {
		REG8_MAPPING.get(index).map(|v| *v)
	}
	
	fn seg_reg(index: usize) -> Option<Reg> {
		SREG_MAPPING.get(index).map(|v| *v)
	}
}

fn treat_u8_as_i8(value: u8) -> i8 {
	unsafe { std::mem::transmute(value) }
}

fn treat_u16_as_i16(value: u16) -> i16 {
	unsafe { std::mem::transmute(value) }
}

fn treat_i8_as_u8(value: i8) -> u8 {
	unsafe { std::mem::transmute(value) }
}

fn treat_i16_as_u16(value: i16) -> u16 {
	unsafe { std::mem::transmute(value) }
}

#[derive(Debug, Clone, PartialEq)]
struct FullReg {
	reg: Reg,
	half: Option<RegHalf>,
}

impl FullReg {
	fn reg(opsize: OpSize, index: usize) -> Option<FullReg> {
		Some(match opsize {
			OpSize::Size8 => {
				let (reg, half) = Reg::reg8(index)?;
				FullReg{reg, half: Some(half)}
			}
			OpSize::Size16 => FullReg{reg: Reg::reg16(index)?, half: None}
		})
	}
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum OpSize {
	Size8,
	Size16,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum OpDirection {
	// This will swap the two arguments of the ModR/M byte, so that the reg bits are the second argument.
	Source,
	Destination,
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum DisplacementOrigin {
	BX_SI,
	BX_DI,
	BP_SI,
	BP_DI,
	SI,
	DI,
	BP,
	BX,
}

// https://www.daenotes.com/electronics/digital-electronics/8086-8088-microprocessor
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
enum Flag {
	// This represents the last bit that carried over the end of a number.
	Carry = 0,
	// This is on if the number of on-bits in the least significant byte of the result of the last
	// operation was even or odd.
	Parity = 2,
	// This means a carry happened between the lower 4 bits and upper 4 bits of an 8 bit number.
	Adjust = 4,
	// This means the result of the last operation was zero.
	Zero = 6,
	// This means the result of the last operation had its most significant bit (the sign bit for
	// signed numbers) set.
	Sign = 7,
	Trap = 8,
	Interrupt = 9,
	Direction = 10,
	// This means the last operation overflowed, eg 255b + 1b causes an overflow.
	Overflow = 11,
}

/// The lookup table for `u8_on_bits_are_odd`.
const U8_ON_BITS_ARE_ODD_LOOKUP: [bool; 256] = [
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	false, true, true, false, true, false, false, true, true, false, false, true, false, true, true, false,
	true, false, false, true, false, true, true, false, false, true, true, false, true, false, false, true,
];

/// This determines if the number of bits that are on in a `u8` value is odd. This is used for
/// `Flag::Parity`.
fn u8_on_bits_are_odd(value: u8) -> bool {
	U8_ON_BITS_ARE_ODD_LOOKUP[value as usize]
}

#[derive(Debug, Copy, Clone, PartialEq, FromPrimitive)]
#[repr(u8)]
enum JumpConditionType {
	Overflow = 0,
	Carry,
	Zero,
	CarryOrZero,
	Sign,
	Parity,
	OverlowIsNotSign,
	OverlowIsNotSignOrZero,
}

#[derive(Debug, Clone, PartialEq)]
struct JumpCondition {
	condition: JumpConditionType,
	negate: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, FromPrimitive)]
#[repr(u8)]
enum ArithmeticMode {
	Add = 0,
	Or,
	AddWithCarry,
	SubWithBorrow,
	And,
	Sub,
	Xor,
	Cmp,
}

#[derive(Debug, Copy, Clone, PartialEq, FromPrimitive)]
#[repr(u8)]
enum ShiftMode {
	RotateLeft = 0,
	RotateRight,
	// "With Carry" means that rather than rotating the bits of the number alone, it adds the
	// carry flag (CF) as the most significant bit while rotating, so for example AL becomes
	// CF:AL (9 bits) and AX becomes CF:AX (17 bits)
	RotateLeftWithCarry,
	RotateRightWithCarry,
	ShiftLeft,
	ShiftRight,
	ShiftLeftArithmethic,
	ShiftRightArithmethic,
}

// https://en.wikibooks.org/wiki/X86_Assembly/Machine_Language_Conversion#Mod_/_Reg_/_R/M_tables
// MOD (2 bits): Register mode
// REG (3 bits): Register
// R/M (3 bits): Register/Memory operand

#[derive(Debug, Copy, Clone, PartialEq)]
enum ModRmRegMode {
	Reg,
	Seg,
	Imm,
}

#[derive(Debug, Clone, PartialEq)]
enum DataLocation8 {
	Reg(Reg, RegHalf),
	AddrSegReg{seg: Reg, reg: Reg},
	//AddrSegRel(Reg, i32),
	AddrDisplacementRel{seg: Reg, displacement: Option<DisplacementOrigin>, offset: i32},
	Immediate(u8),
}

#[derive(Debug, Clone, PartialEq)]
enum DataLocation16 {
	Reg(Reg),
	AddrSegReg{seg: Reg, reg: Reg},
	//AddrSegRel(Reg, i32),
	AddrDisplacementRel{seg: Reg, displacement: Option<DisplacementOrigin>, offset: i32},
	Immediate(u16),
}

#[derive(Debug, Clone, PartialEq)]
enum DataLocation {
	Size8(DataLocation8),
	Size16(DataLocation16),
}

impl DataLocation {
	fn with_immediate_source(self, imm: u16) -> SourceDestination {
		match self {
			DataLocation::Size8(destination) => SourceDestination::Size8(DataLocation8::Immediate(imm as u8), destination),
			DataLocation::Size16(destination) => SourceDestination::Size16(DataLocation16::Immediate(imm), destination),
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
enum SourceDestination {
	Size8(DataLocation8, DataLocation8),
	Size16(DataLocation16, DataLocation16),
}

impl SourceDestination {
	fn split(self) -> (DataLocation, DataLocation) {
		match self {
			SourceDestination::Size8(source, destination) => (DataLocation::Size8(source), DataLocation::Size8(destination)),
			SourceDestination::Size16(source, destination) => (DataLocation::Size16(source), DataLocation::Size16(destination)),
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
enum Inst {
	NoOp,
	Push16(DataLocation16),
	Pop16(DataLocation16),
	Mov(SourceDestination),
	Swap(SourceDestination),
	/// Save as Mov, but increments the Reg by 1 or 2 for 8 and 16 bit SourceDestinations,
	/// respectively. It will do the same to the Option<Reg> if it is also set. If the Direction
	/// flag is set, it will decrement instead.
	// TODO: Maybe this should just match the SourceDestination for Addr locations and increment the
	// associated registers?
	MovAndIncrement(SourceDestination, Reg, Option<Reg>),
	MovReg32{source_h: Reg, source_l: Reg, dest_h: Reg, dest_l: Reg},
	Arithmetic(ArithmeticMode, SourceDestination),
	Inc(DataLocation),
	Dec(DataLocation),
	Rotate{by: DataLocation8, target: DataLocation, mode: ShiftMode},
	//DecBy16(DataLocation16, u16),
	//IncBy16(DataLocation16, u16),
	SetFlag(Flag, bool),
	RepeatNextRegTimes{reg: Reg, until_zero_flag: Option<bool>},
	Call(i32),
	// After popping the IP from the stack, it will pop extra_pop more bytes from the stack. These
	// would probably be arguments that were pushed to the stack just before calling the function in
	// the first place.
	Ret{extra_pop: u16},
	CallAbsolute{ip: u16, cs: u16},
	RetAbsolute,
	Jump{condition: Option<JumpCondition>, offset: i32},
	JumpAbsolute{seg: Reg, displacement: Option<DisplacementOrigin>, offset: i32},
	JumpAndDecrementUntilZero{offset: i32, dec_reg: Reg},
	JumpZeroReg{offset: i32, reg: Reg},
	Interrupt(u8),
	Halt,
	// Override the default segment for the next instruction.
	// https://www.quora.com/Why-is-a-segment-override-prefix-used-with-an-example-in-8086-microprocessor
	//OverrideNextDefaultSegment(Reg)
}

struct Machine8086 {
	memory: Vec<u8>,
	registers: [u16; REG_COUNT],
	halted: bool,
	
	override_default_segment: Option<Reg>,
	number_of_parsed_instructions: usize,
}

impl Machine8086 {
	fn new(memory_bytes: usize) -> Machine8086 {
		let mut machine = Machine8086 {
			memory: vec![0; memory_bytes],
			registers: [0; REG_COUNT as usize],
			halted: false,
			
			override_default_segment: None,
			number_of_parsed_instructions: 0,
		};
		
		machine.set_flag(Flag::Interrupt, true);
		// This bit is set by default in ZETA for some reason.
		machine.registers[Reg::Flags as usize] |= 0b10;
		
		machine
	}
	
	fn insert_contiguous_bytes(&mut self, bytes: &[u8], at: usize) {
		self.memory.splice(at..(at + bytes.len()), bytes.iter().cloned());
	}
	
	fn calculate_displacement(&self, origin: DisplacementOrigin) -> u16 {
		match origin {
			DisplacementOrigin::BX_SI => self.get_reg_u16(Reg::BX).wrapping_add(self.get_reg_u16(Reg::SI)),
			DisplacementOrigin::BX_DI => self.get_reg_u16(Reg::BX).wrapping_add(self.get_reg_u16(Reg::DI)),
			DisplacementOrigin::BP_SI => self.get_reg_u16(Reg::BP).wrapping_add(self.get_reg_u16(Reg::SI)),
			DisplacementOrigin::BP_DI => self.get_reg_u16(Reg::BP).wrapping_add(self.get_reg_u16(Reg::DI)),
			DisplacementOrigin::SI => self.get_reg_u16(Reg::SI),
			DisplacementOrigin::DI => self.get_reg_u16(Reg::DI),
			DisplacementOrigin::BP => self.get_reg_u16(Reg::BP),
			DisplacementOrigin::BX => self.get_reg_u16(Reg::BX),
		}
	}
	
	fn get_data_u8(&self, location: &DataLocation8) -> u8 {
		match location {
			DataLocation8::Reg(reg, half) => {
				self.get_reg_u8(*reg, *half)
			}
			DataLocation8::AddrSegReg{seg, reg} => {
				let origin = self.get_seg_reg(*seg, *reg);
				self.peek_u8(origin)
			}
			/*DataLocation8::AddrSegRel(seg, offset) => {
				let origin = self.get_seg_origin(*seg);
				self.peek_u8((origin as i32 + offset) as u32)
			}*/
			DataLocation8::AddrDisplacementRel{seg, displacement, offset} => {
				let mut origin = self.get_seg_origin(*seg);
				if let Some(displacement) = displacement {
					origin += self.calculate_displacement(*displacement) as u32;
				}
				self.peek_u8(((origin as i32) + *offset) as u32)
			}
			DataLocation8::Immediate(value) => *value,
		}
	}
	
	fn set_data_u8(&mut self, location: &DataLocation8, value: u8) {
		match location {
			DataLocation8::Reg(reg, half) => {
				self.set_reg_u8(*reg, *half, value);
			}
			DataLocation8::AddrSegReg{seg, reg} => {
				let origin = self.get_seg_reg(*seg, *reg);
				self.poke_u8(origin, value);
			}
			/*DataLocation8::AddrSegRel(seg, offset) => {
				let origin = self.get_seg_origin(*seg);
				self.poke_u8((origin as i32 + offset) as u32, value);
			}*/
			DataLocation8::AddrDisplacementRel{seg, displacement, offset} => {
				let mut origin = self.get_seg_origin(*seg);
				if let Some(displacement) = displacement {
					origin += self.calculate_displacement(*displacement) as u32;
				}
				self.poke_u8(((origin as i32) + *offset) as u32, value);
			}
			DataLocation8::Immediate(value) => panic!("Attempted to use immediate as destination: {}", value),
		}
	}

	fn add_to_data_u8(&mut self, location: &DataLocation8, amount: u8) {
		let value = self.get_data_u8(location);
		self.set_data_u8(location, value.wrapping_add(amount));
	}

	fn sub_from_data_u8(&mut self, location: &DataLocation8, amount: u8) {
		let value = self.get_data_u8(location);
		self.set_data_u8(location, value.wrapping_sub(amount));
	}
	
	fn get_data_u16(&self, location: &DataLocation16) -> u16 {
		match location {
			DataLocation16::Reg(reg) => {
				self.get_reg_u16(*reg)
			}
			DataLocation16::AddrSegReg{seg, reg} => {
				let origin = self.get_seg_reg(*seg, *reg);
				self.peek_u16(origin)
			}
			/*DataLocation16::AddrSegRel(seg, offset) => {
				let origin = self.get_seg_origin(*seg);
				//println!("{:?}", (origin as i32 + offset) as u32);
				self.peek_u16((origin as i32 + offset) as u32)
			}*/
			DataLocation16::AddrDisplacementRel{seg, displacement, offset} => {
				let mut origin = self.get_seg_origin(*seg);
				if let Some(displacement) = displacement {
					origin += self.calculate_displacement(*displacement) as u32;
				}
				self.peek_u16(((origin as i32) + *offset) as u32)
			}
			DataLocation16::Immediate(value) => *value,
		}
	}
	
	fn set_data_u16(&mut self, location: &DataLocation16, value: u16) {
		match location {
			DataLocation16::Reg(reg) => {
				self.set_reg_u16(*reg, value);
			}
			DataLocation16::AddrSegReg{seg, reg} => {
				let origin = self.get_seg_reg(*seg, *reg);
				self.poke_u16(origin, value);
			}
			/*DataLocation16::AddrSegRel(seg, offset) => {
				let origin = self.get_seg_origin(*seg);
				self.poke_u16((origin as i32 + offset) as u32, value);
			}*/
			DataLocation16::AddrDisplacementRel{seg, displacement, offset} => {
				let mut origin = self.get_seg_origin(*seg);
				if let Some(displacement) = displacement {
					origin += self.calculate_displacement(*displacement) as u32;
				}
				self.poke_u16(((origin as i32) + *offset) as u32, value);
			}
			DataLocation16::Immediate(value) => panic!("Attempted to use immediate as destination: {}", value),
		}
	}
	
	fn add_to_data_u16(&mut self, location: &DataLocation16, amount: u16) {
		let value = self.get_data_u16(location);
		self.set_data_u16(location, value.wrapping_add(amount));
	}
	
	fn sub_from_data_u16(&mut self, location: &DataLocation16, amount: u16) {
		let value = self.get_data_u16(location);
		self.set_data_u16(location, value.wrapping_sub(amount));
	}
	
	fn get_reg_u8(&self, reg: Reg, half: RegHalf) -> u8 {
		let value16 = self.registers[reg as usize];
		match half {
			RegHalf::High => (value16 >> 8) as u8,
			RegHalf::Low => (value16 & 0xff) as u8,
		}
	}
	
	fn set_reg_u8(&mut self, reg: Reg, half: RegHalf, value: u8) {
		let value16 = &mut self.registers[reg as usize];
		match half {
			RegHalf::High => {
				*value16 = (*value16 & 0x00ff) | ((value as u16) << 8);
			}
			RegHalf::Low => {
				*value16 = (*value16 & 0xff00) | (value as u16);
			}
		}
	}

	fn get_reg_u16(&self, reg: Reg) -> u16 {
		self.registers[reg as usize]
	}
	
	fn set_reg_u16(&mut self, reg: Reg, value: u16) {
		self.registers[reg as usize] = value;
	}
	
	fn add_to_reg(&mut self, reg: Reg, amount: u16) {
		let reg_value = &mut self.registers[reg as usize];
		*reg_value = reg_value.wrapping_add(amount);
	}
	
	fn sub_from_reg(&mut self, reg: Reg, amount: u16) {
		let reg_value = &mut self.registers[reg as usize];
		*reg_value = reg_value.wrapping_sub(amount);
	}

	fn get_seg_origin(&self, seg_reg: Reg) -> u32 {
		((self.registers[seg_reg as usize] as u32) << 4) & 0xFFFFF
	}

	fn get_seg_reg(&self, seg_reg: Reg, reg: Reg) -> u32 {
		(((self.registers[seg_reg as usize] as u32) << 4) + self.registers[reg as usize] as u32) & 0xFFFFF
	}
	
	fn get_ip(&self) -> u32 {
		self.get_seg_reg(Reg::CS, Reg::IP)
	}
	
	fn get_sp(&self) -> u32 {
		self.get_seg_reg(Reg::SS, Reg::SP)
	}
	
	fn peek_u8(&self, at: u32) -> u8 {
		self.memory[at as usize]
	}
	
	fn poke_u8(&mut self, at: u32, value: u8) {
		self.memory[at as usize] = value;
	}
	
	fn peek_u16(&self, at: u32) -> u16 {
		((self.memory[at as usize + 1] as u16) << 8) + self.memory[at as usize] as u16
	}
	
	fn poke_u16(&mut self, at: u32, value: u16) {
		self.memory[at as usize] = (value & 0x00ff) as u8;
		self.memory[at as usize + 1] = ((value & 0xff00) >> 8) as u8;
	}
	
	fn push_u16(&mut self, value: u16) {
		println!("Push16({})", value);
		self.sub_from_reg(Reg::SP, 2);
		self.poke_u16(self.get_sp(), value);
	}
	
	fn pop_u16(&mut self) -> u16 {
		let value = self.peek_u16(self.get_sp());
		println!("Pop16({})", value);
		self.add_to_reg(Reg::SP, 2);
		value
	}
	
	fn read_ip_u8(&mut self) -> u8 {
		let value = self.peek_u8(self.get_ip());
		self.add_to_reg(Reg::IP, 1);
		value
	}
	
	fn read_ip_u16(&mut self) -> u16 {
		let value = self.peek_u16(self.get_ip());
		self.add_to_reg(Reg::IP, 2);
		value
	}
	
	fn set_flag(&mut self, flag: Flag, on: bool) {
		let mut flags = self.get_reg_u16(Reg::Flags);
		let bit = 0b1 << (flag as u16);
		if on {
			flags = flags | bit;
		} else {
			flags = flags & (!bit);
		}
		self.set_reg_u16(Reg::Flags, flags);
	}
	
	fn get_flag(&self, flag: Flag) -> bool {
		let flags = self.get_reg_u16(Reg::Flags);
		let bit = 0b1 << (flag as u16);
		(flags & bit) != 0
	}
	
	fn test_jump_condition(&self, condition: &JumpCondition) -> bool {
		let pos_result = match condition.condition {
			JumpConditionType::Overflow => self.get_flag(Flag::Overflow),
			JumpConditionType::Carry => self.get_flag(Flag::Carry),
			JumpConditionType::Zero => self.get_flag(Flag::Zero),
			JumpConditionType::CarryOrZero => self.get_flag(Flag::Carry) || self.get_flag(Flag::Zero),
			JumpConditionType::Sign => self.get_flag(Flag::Sign),
			JumpConditionType::Parity => self.get_flag(Flag::Parity),
			JumpConditionType::OverlowIsNotSign => self.get_flag(Flag::Overflow) != self.get_flag(Flag::Sign),
			JumpConditionType::OverlowIsNotSignOrZero => self.get_flag(Flag::Overflow) != self.get_flag(Flag::Sign) || self.get_flag(Flag::Zero),
		};
		if condition.negate { !pos_result } else { pos_result }
	}
	
	fn resolve_default_segment(&self, default_segment: Reg) -> Reg {
		if let Some(override_default_segment) = self.override_default_segment {
			override_default_segment
		} else {
			default_segment
		}
	}
	
	fn read_modrm_source_destination(&mut self, opsize: OpSize, opdir: OpDirection, reg_mode: ModRmRegMode, default_segment: Reg, override_mod: Option<u8>) -> SourceDestination {
		let modrm_code = self.read_ip_u8();
		//println!("Modrm code: {:b}", modrm_code);
		let rm_code = modrm_code & 0b00000111;
		let reg_code = (modrm_code & 0b00111000) >> 3;
		let mod_code = if let Some(override_mod) = override_mod {
			override_mod
		} else {
			(modrm_code & 0b11000000) >> 6
		};
		
		let mut first: DataLocation = match reg_mode {
			ModRmRegMode::Reg => match opsize {
				OpSize::Size8 => {
					let (reg, half) = Reg::reg8(reg_code as usize).unwrap();
					DataLocation::Size8(DataLocation8::Reg(reg, half))
				}
				OpSize::Size16 => DataLocation::Size16(DataLocation16::Reg(Reg::reg16(reg_code as usize).unwrap())),
			}
			ModRmRegMode::Seg => match opsize {
				OpSize::Size8 => panic!("Reg cannot be seg in 8 bit mode"),
				OpSize::Size16 => DataLocation::Size16(DataLocation16::Reg(Reg::seg_reg(reg_code as usize).unwrap())),
			}
			ModRmRegMode::Imm => match opsize {
				OpSize::Size8 => DataLocation::Size8(DataLocation8::Immediate(reg_code)),
				OpSize::Size16 => DataLocation::Size16(DataLocation16::Immediate(reg_code as u16)),
			}
		};
		
		let mut second: Option<DataLocation> = None;
		let mut displacement: Option<i32> = None;
		
		let displacement_origin = match rm_code {
			0b000 => DisplacementOrigin::BX_SI,
			0b001 => DisplacementOrigin::BX_DI,
			0b010 => DisplacementOrigin::BP_SI,
			0b011 => DisplacementOrigin::BP_DI,
			0b100 => DisplacementOrigin::SI,
			0b101 => DisplacementOrigin::DI,
			0b110 => DisplacementOrigin::BP,
			0b111 => DisplacementOrigin::BX,
			_ => unreachable!()
		};
		
		match mod_code {
			0b00 => {
				// If r/m is 110, Displacement (16 bits) is address; otherwise, no displacement
				if rm_code == 0b110 {
					let addr = self.read_ip_u16() as i32;
					// Displacements are relative to the data segment (DS)
					let seg = self.resolve_default_segment(default_segment);
					second = Some(match opsize {
						OpSize::Size8 => DataLocation::Size8(DataLocation8::AddrDisplacementRel{seg, displacement: None, offset: addr}),
						OpSize::Size16 => DataLocation::Size16(DataLocation16::AddrDisplacementRel{seg, displacement: None, offset: addr}),
					});
				} else {
					displacement = Some(0);
				};
				
			}
			0b01 => {
				// Eight-bit displacement, sign-extended to 16 bits
				displacement = Some(treat_u8_as_i8(self.read_ip_u8()) as i32);
			}
			0b10 => {
				// 16-bit displacement (example: MOV [BX + SI]+ displacement,al)
				displacement = Some(treat_u16_as_i16(self.read_ip_u16()) as i32);
			}
			0b11 => {
				// r/m is treated as a second "reg" field
				second = Some(match opsize {
					OpSize::Size8 => {
						let (reg, half) = Reg::reg8(rm_code as usize).unwrap();
						DataLocation::Size8(DataLocation8::Reg(reg, half))
					}
					OpSize::Size16 => DataLocation::Size16(DataLocation16::Reg(Reg::reg16(rm_code as usize).unwrap()))
				});
			}
			_ => unreachable!()
		}
		
		if let Some(offset) = displacement {
			let seg = self.resolve_default_segment(default_segment);
			second = Some(match opsize {
				OpSize::Size8 => DataLocation::Size8(DataLocation8::AddrDisplacementRel{seg, displacement: Some(displacement_origin), offset}),
				OpSize::Size16 => DataLocation::Size16(DataLocation16::AddrDisplacementRel{seg, displacement: Some(displacement_origin), offset}),
			});
		}
		
		match (first, second) {
			(DataLocation::Size8(source), Some(DataLocation::Size8(destination))) => {
				match opdir {
					OpDirection::Source => SourceDestination::Size8(source, destination),
					OpDirection::Destination => SourceDestination::Size8(destination, source),
				}
			}
			(DataLocation::Size16(source), Some(DataLocation::Size16(destination))) => {
				match opdir {
					OpDirection::Source => SourceDestination::Size16(source, destination),
					OpDirection::Destination => SourceDestination::Size16(destination, source),
				}
			}
			_ => panic!("ModR/M byte didn't fully resolve")
		}
	}
	
	fn read_modrm_with_immediate_reg_u8(&mut self, default_segment: Reg) -> (u8, DataLocation8) {
		let source_destination = self.read_modrm_source_destination(OpSize::Size8, OpDirection::Source, ModRmRegMode::Imm, default_segment, None);
		let (imm, destination) = match source_destination {
			SourceDestination::Size8(source, destination) => match source {
				DataLocation8::Immediate(imm) => (imm, destination),
				_ => panic!("Source was not an immediate value"),
			}
			SourceDestination::Size16(_, _) => panic!("Source was not 8 bits"),
		};
		(imm, destination)
	}
	
	fn read_modrm_with_immediate_reg_u16(&mut self, default_segment: Reg, override_mod: Option<u8>) -> (u16, DataLocation16) {
		let source_destination = self.read_modrm_source_destination(OpSize::Size16, OpDirection::Source, ModRmRegMode::Imm, default_segment, override_mod);
		let (imm, destination) = match source_destination {
			SourceDestination::Size8(_, _) => panic!("Source was not 16 bits"),
			SourceDestination::Size16(source, destination) => match source {
				DataLocation16::Immediate(imm) => (imm, destination),
				_ => panic!("Source was not an immediate value"),
			}
		};
		(imm, destination)
	}
	
	fn read_standard_source_destination(&mut self, opcode: u8, reg_mode: ModRmRegMode, default_segment: Reg) -> SourceDestination {
		let opsize = if opcode & 0b01 == 0 { OpSize::Size8 } else { OpSize::Size16 };
		let opdir = if opcode & 0b10 == 0 { OpDirection::Source } else { OpDirection::Destination };
		self.read_modrm_source_destination(opsize, opdir, reg_mode, default_segment, None)
	}
	
	fn read_arithmetic_source_destination_with_ax(&mut self, opcode: u8) -> SourceDestination {
		match opcode & 0b111 {
			0x00 ... 0x03 => self.read_standard_source_destination(opcode, ModRmRegMode::Reg, Reg::DS),
			0x04 => {
				let imm = self.read_ip_u8();
				SourceDestination::Size8(DataLocation8::Immediate(imm), DataLocation8::Reg(Reg::AX, RegHalf::Low))
			}
			0x05 => {
				let imm = self.read_ip_u16();
				SourceDestination::Size16(DataLocation16::Immediate(imm), DataLocation16::Reg(Reg::AX))
			}
			_ => panic!("Attempted to call read_arithmetic_source_destination with bad opcode: 0x{:x}", opcode)
		}
	}

	fn parse_instruction(&mut self) -> Inst {
		let opcode = self.read_ip_u8();
		//println!("{:?}", self.registers);
		println!("Opcode: 0x{:02x} ({:?})", opcode, self.number_of_parsed_instructions);
		self.number_of_parsed_instructions += 1;
		// 673096
		/*if self.number_of_parsed_instructions == 673099 {
			panic!();
		}*/
		//println!("IP: {:?}", self.get_ip());
		match opcode {
			0x00 ... 0x05 => {
				let inst = Inst::Arithmetic(ArithmeticMode::Add, self.read_arithmetic_source_destination_with_ax(opcode));
				if opcode == 0x01 {
					println!("{:?}", inst);
					println!("{:?}, {:?}", self.get_reg_u16(Reg::AX), self.get_reg_u16(Reg::DS));
				}
				inst
			}
			0x06 => Inst::Push16(DataLocation16::Reg(Reg::ES)),
			0x08 ... 0x0d => Inst::Arithmetic(ArithmeticMode::Or, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x0e => Inst::Push16(DataLocation16::Reg(Reg::CS)),
			0x1e => Inst::Push16(DataLocation16::Reg(Reg::DS)),
			0x1f => Inst::Pop16(DataLocation16::Reg(Reg::DS)),
			0x26 => {
				self.override_default_segment = Some(Reg::ES);
				let inst = self.parse_instruction();
				self.override_default_segment = None;
				inst
			}
			0x28 ... 0x2d => Inst::Arithmetic(ArithmeticMode::Sub, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x2e => {
				self.override_default_segment = Some(Reg::CS);
				let inst = self.parse_instruction();
				self.override_default_segment = None;
				inst
			}
			0x30 ... 0x35 => Inst::Arithmetic(ArithmeticMode::Xor, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x38 ... 0x3d => Inst::Arithmetic(ArithmeticMode::Cmp, self.read_arithmetic_source_destination_with_ax(opcode)),
			0x3c => {
				let imm = self.read_ip_u8();
				Inst::Arithmetic(ArithmeticMode::Cmp, SourceDestination::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low), DataLocation8::Immediate(imm)))
			}
			0x3d => {
				let imm = self.read_ip_u16();
				Inst::Arithmetic(ArithmeticMode::Cmp, SourceDestination::Size16(DataLocation16::Reg(Reg::AX), DataLocation16::Immediate(imm)))
			}
			0x40 ... 0x47 => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				//Inst::IncBy16(DataLocation16::Reg(reg), 1)
				Inst::Inc(DataLocation::Size16(DataLocation16::Reg(reg)))
			}
			0x48 ... 0x4f => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				//Inst::DecBy16(DataLocation16::Reg(reg), 1)
				Inst::Dec(DataLocation::Size16(DataLocation16::Reg(reg)))
			}
			0x50 ... 0x57 => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				Inst::Push16(DataLocation16::Reg(reg))
			}
			0x58 ... 0x5f => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				Inst::Pop16(DataLocation16::Reg(reg))
			}
			0x70 ... 0x7f => {
				let double_condition = opcode - 0x70;
				let condition_type = JumpConditionType::from_u8(double_condition >> 1).unwrap();
				let negate = double_condition % 2 == 1;
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::Jump{condition: Some(JumpCondition{condition: condition_type, negate}), offset}
			}
			0x80 | 0x82 => {
				let (mode_index, destination) = self.read_modrm_with_immediate_reg_u8(Reg::DS);
				let arithmetic_mode = ArithmeticMode::from_u8(mode_index).unwrap();
				let imm = self.read_ip_u8();
				Inst::Arithmetic(arithmetic_mode, SourceDestination::Size8(DataLocation8::Immediate(imm), destination))
			}
			0x81 => {
				let (mode_index, destination) = self.read_modrm_with_immediate_reg_u16(Reg::DS, None);
				let arithmetic_mode = ArithmeticMode::from_u16(mode_index).unwrap();
				let imm = self.read_ip_u16();
				Inst::Arithmetic(arithmetic_mode, SourceDestination::Size16(DataLocation16::Immediate(imm), destination))
			}
			// 0x82: See 0x80
			0x83 => {
				let (mode_index, destination) = self.read_modrm_with_immediate_reg_u16(Reg::DS, None);
				let arithmetic_mode = ArithmeticMode::from_u16(mode_index).unwrap();
				let raw_imm = self.read_ip_u8();
				// Read a u8 then sign-extend it to u16:
				let imm = treat_i16_as_u16(treat_u8_as_i8(raw_imm) as i16);
				println!("0x83 {}, {}", raw_imm, imm);
				Inst::Arithmetic(arithmetic_mode, SourceDestination::Size16(DataLocation16::Immediate(imm as u16), destination))
			}
			0x88 ... 0x8b => Inst::Mov(self.read_standard_source_destination(opcode, ModRmRegMode::Reg, Reg::DS)),
			0x8c => Inst::Mov(self.read_modrm_source_destination(OpSize::Size16, OpDirection::Source, ModRmRegMode::Seg, Reg::DS, None)),
			0x8e => Inst::Mov(self.read_modrm_source_destination(OpSize::Size16, OpDirection::Destination, ModRmRegMode::Seg, Reg::DS, None)),
			0x90 ... 0x97 => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				Inst::Swap(SourceDestination::Size16(DataLocation16::Reg(reg), DataLocation16::Reg(Reg::AX)))
			}
			// CALLF
			0x9a => {
				let ip = self.read_ip_u16();
				let cs = self.read_ip_u16();
				Inst::CallAbsolute{ip, cs}
			},
			0xa1 => {
				let addr = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size16(DataLocation16::AddrDisplacementRel{seg: self.resolve_default_segment(Reg::DS), displacement: None, offset: addr as i32}, DataLocation16::Reg(Reg::AX)))
			}
			0xa2 => {
				let addr = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low), DataLocation8::AddrDisplacementRel{seg: self.resolve_default_segment(Reg::DS), displacement: None, offset: addr as i32}))
			}
			0xa3 => {
				let addr = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size16(DataLocation16::Reg(Reg::AX), DataLocation16::AddrDisplacementRel{seg: self.resolve_default_segment(Reg::DS), displacement: None, offset: addr as i32}))
			}
			// MOVSB 
			// "string" means that it increments (or decrements if the direction flag is set) the
			// memory address register(s) after doing an operation.
			0xa4 => Inst::MovAndIncrement(SourceDestination::Size8(DataLocation8::AddrSegReg{seg: Reg::DS, reg: Reg::SI}, DataLocation8::AddrSegReg{seg: Reg::ES, reg: Reg::DI}), Reg::SI, Some(Reg::DI)),
			0xa5 => Inst::MovAndIncrement(SourceDestination::Size16(DataLocation16::AddrSegReg{seg: Reg::DS, reg: Reg::SI}, DataLocation16::AddrSegReg{seg: Reg::ES, reg: Reg::DI}), Reg::SI, Some(Reg::DI)),
			// STOSB
			0xaa => Inst::MovAndIncrement(SourceDestination::Size8(DataLocation8::Reg(Reg::AX, RegHalf::Low), DataLocation8::AddrSegReg{seg: Reg::ES, reg: Reg::DI}), Reg::DI, None),
			// STOSW
			0xab => Inst::MovAndIncrement(SourceDestination::Size16(DataLocation16::Reg(Reg::AX), DataLocation16::AddrSegReg{seg: Reg::ES, reg: Reg::DI}), Reg::DI, None),
			// LODSB
			0xac => Inst::MovAndIncrement(SourceDestination::Size8(DataLocation8::AddrSegReg{seg: Reg::DS, reg: Reg::SI}, DataLocation8::Reg(Reg::AX, RegHalf::Low)), Reg::SI, None),
			// LODSW
			0xad => Inst::MovAndIncrement(SourceDestination::Size16(DataLocation16::AddrSegReg{seg: Reg::DS, reg: Reg::SI}, DataLocation16::Reg(Reg::AX)), Reg::SI, None),
			0xb0 ... 0xb7 => {
				let (reg, reg_half) = Reg::reg8((opcode & 0b111) as usize).unwrap();
				let imm = self.read_ip_u8();
				Inst::Mov(SourceDestination::Size8(DataLocation8::Immediate(imm), DataLocation8::Reg(reg, reg_half)))
			}
			0xb8 ... 0xbf => {
				let reg = Reg::reg16((opcode & 0b111) as usize).unwrap();
				let imm = self.read_ip_u16();
				Inst::Mov(SourceDestination::Size16(DataLocation16::Immediate(imm), DataLocation16::Reg(reg)))
			}
			0xc0 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u8(Reg::DS);
				let shift_mode = ShiftMode::from_u8(shift_index).unwrap();
				let imm = self.read_ip_u8();
				Inst::Rotate{by: DataLocation8::Immediate(imm), target: DataLocation::Size8(destination), mode: shift_mode}
			}
			0xc1 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u16(Reg::DS, None);
				let shift_mode = ShiftMode::from_u16(shift_index).unwrap();
				let imm = self.read_ip_u8();
				Inst::Rotate{by: DataLocation8::Immediate(imm), target: DataLocation::Size16(destination), mode: shift_mode}
			}
			// RETN
			0xc2 => {
				let extra_pop = self.read_ip_u16();
				Inst::Ret{extra_pop}
			}
			// RETN
			0xc3 => {
				Inst::Ret{extra_pop: 0}
			}
			// LES (load absolute pointer in the ES segment)
			0xc4 => {
				let source_destination = self.read_modrm_source_destination(OpSize::Size16, OpDirection::Destination, ModRmRegMode::Imm, Reg::DS, None);
				match source_destination {
					SourceDestination::Size16(source, destination) => {
						let source_value = self.get_data_u16(&source);
						panic!("c4: {:?} -> {:?} ({:?})", source, destination, source_value);
						//self.resolve_default_segment(source)
					}
					_ => panic!("Expected 16-bit result")
				}
			}
			0xc6 => {
				let (_, destination) = self.read_modrm_source_destination(OpSize::Size8, OpDirection::Destination, ModRmRegMode::Reg, Reg::DS, None).split();
				let imm = self.read_ip_u8();
				Inst::Mov(destination.with_immediate_source(imm as u16))
			}
			0xc7 => {
				let (_, destination) = self.read_modrm_source_destination(OpSize::Size16, OpDirection::Destination, ModRmRegMode::Reg, Reg::DS, None).split();
				let imm = self.read_ip_u16();
				Inst::Mov(destination.with_immediate_source(imm))
			}
			0xcb => Inst::RetAbsolute,
			0xcd => {
				let interrupt_index = self.read_ip_u8();
				Inst::Interrupt(interrupt_index)
			}
			0xd0 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u8(Reg::DS);
				let shift_mode = ShiftMode::from_u8(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Immediate(1), target: DataLocation::Size8(destination), mode: shift_mode}
			}
			0xd1 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u16(Reg::DS, None);
				let shift_mode = ShiftMode::from_u16(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Immediate(1), target: DataLocation::Size16(destination), mode: shift_mode}
			}
			0xd2 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u8(Reg::DS);
				let shift_mode = ShiftMode::from_u8(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Reg(Reg::CX, RegHalf::Low), target: DataLocation::Size8(destination), mode: shift_mode}
			}
			0xd3 => {
				let (shift_index, destination) = self.read_modrm_with_immediate_reg_u16(Reg::DS, None);
				let shift_mode = ShiftMode::from_u16(shift_index).unwrap();
				Inst::Rotate{by: DataLocation8::Reg(Reg::CX, RegHalf::Low), target: DataLocation::Size16(destination), mode: shift_mode}
			}
			// LOOP
			0xe2 => {
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::JumpAndDecrementUntilZero{offset, dec_reg: Reg::CX}
			}
			0xe3 => {
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::JumpZeroReg{offset, reg: Reg::CX}
			}
			0xe8 => {
				let offset = treat_u16_as_i16(self.read_ip_u16()) as i32;
				Inst::Call(offset)
			}
			0xe9 => {
				let offset = treat_u16_as_i16(self.read_ip_u16()) as i32;
				Inst::Jump{condition: None, offset}
			}
			0xeb => {
				let offset = treat_u8_as_i8(self.read_ip_u8()) as i32;
				Inst::Jump{condition: None, offset}
			}
			// REPZ
			0xf3 => Inst::RepeatNextRegTimes{reg: Reg::CX, until_zero_flag: Some(true)},
			0xf4 => Inst::Halt,
			0xf8 => Inst::SetFlag(Flag::Carry, false),
			0xf9 => Inst::SetFlag(Flag::Carry, true),
			0xfa => Inst::SetFlag(Flag::Interrupt, false),
			0xfb => Inst::SetFlag(Flag::Interrupt, true),
			0xfc => Inst::SetFlag(Flag::Direction, false),
			0xfd => Inst::SetFlag(Flag::Direction, true),
			0xff => {
				let (inst_index, destination) = self.read_modrm_with_immediate_reg_u16(Reg::DS, None);
				match inst_index {
					5 => {
						// JMPF (jump far)
						// This gets the modrm destination (which should be a seg/offset for getting
						// a memory address) and reads 32 bits from that position. The first 16 bits
						// go into the IP and the second 16 go into the CS.
						if let DataLocation16::AddrDisplacementRel{seg, displacement, offset} = destination {
							Inst::JumpAbsolute{seg, displacement, offset}
						} else {
							panic!("Expected AddrDisplacementRel for JMPF: {:?}", destination);
						}
					}
					_ => panic!("Unknown 0xff mode: {:?}", inst_index)
				}
			}
			_ => {
				//println!("{}", String::from_utf8_lossy(&self.memory));
				panic!("Unknown opcode: 0x{:02x}", opcode);
			}
		}
	}
	
	/// Returns a value if the destination is meant to be overwritten (eg. a cmp doesn't overwrite
	/// the destination, so it would return None).
	fn apply_arithmetic_inst<ValueType: AnyUnsignedInt8086>(&mut self, value1: ValueType, value2: ValueType, mode: ArithmeticMode, change_carry_flag: bool) -> Option<ValueType> {
		enum CarryType {
			Clear,
			Add,
			Sub,
			SubWithBorrow,
		}
		let (result_value, should_write, set_adjust_flag, carry_type) = match mode {
			ArithmeticMode::Add => (value1.wrapping_add(value2), true, true, CarryType::Add),
			ArithmeticMode::Or => (value1 | value2, true, false, CarryType::Clear),
			ArithmeticMode::AddWithCarry => {
				let carry_value = if self.get_flag(Flag::Carry) { ValueType::ONE } else { ValueType::ZERO };
				(value1.wrapping_add(value2).wrapping_add(carry_value), true, true, CarryType::Add)
			}
			ArithmeticMode::SubWithBorrow => {
				let carry_value = if self.get_flag(Flag::Carry) { ValueType::ONE } else { ValueType::ZERO };
				(value1.wrapping_sub(value2.wrapping_add(carry_value)), true, true, CarryType::SubWithBorrow)
			}
			ArithmeticMode::And => (value1 & value2, true, false, CarryType::Clear),
			ArithmeticMode::Sub => (value1.wrapping_sub(value2), true, true, CarryType::Sub),
			ArithmeticMode::Xor => (value1 ^ value2, true, false, CarryType::Clear),
			ArithmeticMode::Cmp => (value1.wrapping_sub(value2), false, true, CarryType::Sub),
		};
		
		self.set_standard_zero_sign_partiy_flags(result_value);
		
		if set_adjust_flag {
			self.set_flag(Flag::Adjust, (((value1 ^ value2) ^ result_value) & (ValueType::ONE << 4)) != ValueType::ZERO);
		} else {
			self.set_flag(Flag::Adjust, false);
		}
		
		let sign_bit = ValueType::ONE << (ValueType::BIT_SIZE - 1);
		
		match carry_type {
			CarryType::Clear => {
				self.set_flag(Flag::Overflow, false);
				self.set_flag(Flag::Carry, false);
			}
			CarryType::Add => {
				self.set_flag(Flag::Overflow, (((value1 ^ value2 ^ sign_bit) & (result_value ^ value2)) & sign_bit) != ValueType::ZERO);
				// If the result value is supposed to increase but it decreases, the carry flag is set.
				if change_carry_flag { self.set_flag(Flag::Carry, result_value < value1); }
			}
			CarryType::Sub => {
				self.set_flag(Flag::Overflow, (((value1 ^ value2) & (value1 ^ result_value)) & sign_bit) != ValueType::ZERO);
				if change_carry_flag { self.set_flag(Flag::Carry, value1 < value2); }
			}
			CarryType::SubWithBorrow => {
				self.set_flag(Flag::Overflow, (((value1 ^ value2) & (value1 ^ result_value)) & sign_bit) != ValueType::ZERO);
				if change_carry_flag { self.set_flag(Flag::Carry, value1 < result_value); }
			}
		}
		
		if should_write { Some(result_value) } else { None }
	}
	
	fn apply_rotate_inst<ValueType: AnyUnsignedInt8086>(&mut self, rotate_amount: u8, mut value: ValueType, mode: ShiftMode) -> ValueType {
		// https://www.felixcloutier.com/x86/rcl:rcr:rol:ror
		// https://www.felixcloutier.com/x86/sal:sar:shl:shr
		let masked_rotate_amount = rotate_amount & 0b11111;
		let real_rotate_amount = (masked_rotate_amount) % ValueType::BIT_SIZE;
		let carry_rotate_amount = (masked_rotate_amount) % (ValueType::BIT_SIZE + 1);
		match mode {
			ShiftMode::RotateLeft => { // ROL
				value = value.rotate_left(real_rotate_amount as u32);
				if masked_rotate_amount != 0 {
					self.set_flag(Flag::Carry, value.least_significant_bit(0));
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != value.least_significant_bit(0));
				}
			}
			ShiftMode::RotateRight => { // ROR
				value = value.rotate_right(real_rotate_amount as u32);
				if masked_rotate_amount != 0 {
					self.set_flag(Flag::Carry, value.most_significant_bit(0));
				}
				if masked_rotate_amount == 1 {
					// The most signficant bit != the second most significant bit.
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != value.most_significant_bit(1));
				}
			}
			ShiftMode::RotateLeftWithCarry => { // RCL
				for _ in 0..carry_rotate_amount {
					let original_most_sig_bit = value.most_significant_bit(0);
					value = value << 1;
					if self.get_flag(Flag::Carry) {
						value += ValueType::ONE;
					}
					self.set_flag(Flag::Carry, original_most_sig_bit);
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != self.get_flag(Flag::Carry));
				}
			}
			ShiftMode::RotateRightWithCarry => { // RCR
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != self.get_flag(Flag::Carry));
				}
				for _ in 0..carry_rotate_amount {
					let original_least_sig_bit = value.least_significant_bit(0);
					value = value >> 1;
					if self.get_flag(Flag::Carry) {
						value |= ValueType::ONE << (ValueType::BIT_SIZE - 1);
					}
					self.set_flag(Flag::Carry, original_least_sig_bit);
				}
			}
			ShiftMode::ShiftLeft | ShiftMode::ShiftLeftArithmethic => { // SHL
				if rotate_amount != 0 {
					value = value << rotate_amount - 1;
					self.set_flag(Flag::Carry, value.most_significant_bit(0));
					value = value << 1;
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, value.most_significant_bit(0) != self.get_flag(Flag::Carry));
				}
				self.set_flag(Flag::Adjust, (rotate_amount & 0x1f) != 0);
			}
			ShiftMode::ShiftRight => { // SHR
				let original_value = value;
				if rotate_amount != 0 {
					value = value >> rotate_amount - 1;
					self.set_flag(Flag::Carry, value.least_significant_bit(0));
					value = value >> 1;
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, original_value.most_significant_bit(0));
				}
			}
			// SAL: ShiftMode::ShiftLeftArithmethic is the same as ShiftMode::ShiftLeft
			ShiftMode::ShiftRightArithmethic => { // SAR
				if rotate_amount != 0 {
					value = value >> rotate_amount - 1;
					self.set_flag(Flag::Carry, value.least_significant_bit(0));
					value = value >> 1;
				}
				if masked_rotate_amount == 1 {
					self.set_flag(Flag::Overflow, false);
				}
			}
		};
		
		self.set_standard_zero_sign_partiy_flags(value);
		
		value
	}
	
	fn set_standard_zero_sign_partiy_flags<ValueType: AnyUnsignedInt8086>(&mut self, value: ValueType) {
		self.set_flag(Flag::Zero, value == ValueType::ZERO);
		self.set_flag(Flag::Sign, value.most_significant_bit(0));
		self.set_flag(Flag::Parity, u8_on_bits_are_odd(value.least_significant_byte()));
	}
	
	fn interrupt(&mut self, interrupt_index: u8) {
		let interrupt_addr = (interrupt_index as u32) * 4;
		let ip = self.peek_u16(interrupt_addr);
		let cs = self.peek_u16(interrupt_addr + 2);
		
		self.push_u16(self.get_reg_u16(Reg::Flags));
		self.push_u16(self.get_reg_u16(Reg::CS));
		self.push_u16(self.get_reg_u16(Reg::IP));
		
		self.set_reg_u16(Reg::IP, ip);
		self.set_reg_u16(Reg::CS, cs);
		self.halted = false;
		self.set_flag(Flag::Interrupt, false);
	}
	
	fn apply_instruction(&mut self, inst: &Inst) {
		match *inst {
			Inst::NoOp => {}
			Inst::Push16(ref location) => {
				self.push_u16(self.get_data_u16(location));
			}
			Inst::Pop16(ref location) => {
				let value = self.pop_u16();
				self.set_data_u16(location, value);
			}
			Inst::Mov(SourceDestination::Size8(ref source, ref destination)) => {
				let value = self.get_data_u8(&source);
				self.set_data_u8(&destination, value);
			}
			Inst::Mov(SourceDestination::Size16(ref source, ref destination)) => {
				let value = self.get_data_u16(&source);
				self.set_data_u16(&destination, value);
			}
			Inst::Swap(SourceDestination::Size8(ref left, ref right)) => {
				let value_left = self.get_data_u8(&left);
				let value_right = self.get_data_u8(&right);
				self.set_data_u8(&left, value_right);
				self.set_data_u8(&right, value_left);
			}
			Inst::Swap(SourceDestination::Size16(ref source, ref destination)) => {
				let value = self.get_data_u16(&source);
				self.set_data_u16(&destination, value);
			}
			Inst::MovAndIncrement(SourceDestination::Size8(ref source, ref destination), inc_reg, inc_reg2) => {
				let value = self.get_data_u8(&source);
				self.set_data_u8(&destination, value);
				if self.get_flag(Flag::Direction) {
					self.sub_from_reg(inc_reg, 1);
					if let Some(inc_reg2) = inc_reg2 { self.sub_from_reg(inc_reg2, 1); }
				} else {
					self.add_to_reg(inc_reg, 1);
					if let Some(inc_reg2) = inc_reg2 { self.add_to_reg(inc_reg2, 1); }
				}
			}
			Inst::MovAndIncrement(SourceDestination::Size16(ref source, ref destination), inc_reg, inc_reg2) => {
				let value = self.get_data_u16(&source);
				self.set_data_u16(&destination, value);
				if self.get_flag(Flag::Direction) {
					self.sub_from_reg(inc_reg, 2);
					if let Some(inc_reg2) = inc_reg2 { self.sub_from_reg(inc_reg2, 2); }
				} else {
					self.add_to_reg(inc_reg, 2);
					if let Some(inc_reg2) = inc_reg2 { self.add_to_reg(inc_reg2, 2); }
				}
			}
			Inst::Arithmetic(mode, SourceDestination::Size8(ref source, ref destination)) => {
				let value1 = self.get_data_u8(&destination);
				let value2 = self.get_data_u8(&source);
				if let Some(result_value) = self.apply_arithmetic_inst(value1, value2, mode, true) {
					self.set_data_u8(&destination, result_value);
				}
			}
			Inst::Arithmetic(mode, SourceDestination::Size16(ref source, ref destination)) => {
				let value1 = self.get_data_u16(&destination);
				let value2 = self.get_data_u16(&source);
				if let Some(result_value) = self.apply_arithmetic_inst(value1, value2, mode, true) {
					self.set_data_u16(&destination, result_value);
				}
			}
			Inst::Inc(DataLocation::Size8(ref target)) => {
				let value = self.get_data_u8(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Add, false) {
					self.set_data_u8(&target, result_value);
				}
			}
			Inst::Inc(DataLocation::Size16(ref target)) => {
				let value = self.get_data_u16(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Add, false) {
					self.set_data_u16(&target, result_value);
				}
			}
			Inst::Dec(DataLocation::Size8(ref target)) => {
				let value = self.get_data_u8(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Sub, false) {
					self.set_data_u8(&target, result_value);
				}
			}
			Inst::Dec(DataLocation::Size16(ref target)) => {
				let value = self.get_data_u16(&target);
				if let Some(result_value) = self.apply_arithmetic_inst(value, 1, ArithmeticMode::Sub, false) {
					self.set_data_u16(&target, result_value);
				}
			}
			Inst::Rotate{ref by, target: DataLocation::Size8(ref target), mode} => {
				let rotate_amount = self.get_data_u8(by);
				let value = self.get_data_u8(&target);
				let new_value = self.apply_rotate_inst(rotate_amount, value, mode);
				self.set_data_u8(&target, new_value);
			}
			Inst::Rotate{ref by, target: DataLocation::Size16(ref target), mode} => {
				let rotate_amount = self.get_data_u8(by);
				let value = self.get_data_u16(&target);
				let new_value = self.apply_rotate_inst(rotate_amount, value, mode);
				self.set_data_u16(&target, new_value);
			}
			/*Inst::DecBy16(location, amount) => {
				self.sub_from_data_u16(&location, amount);
			}
			Inst::IncBy16(location, amount) => {
				self.add_to_data_u16(&location, amount);
			}*/
			Inst::SetFlag(flag, on) => {
				self.set_flag(flag, on);
			}
			Inst::RepeatNextRegTimes{reg, until_zero_flag} => {
				let repeat_inst = self.parse_instruction();
				dbg!(self.get_reg_u16(reg));
				while self.get_reg_u16(reg) != 0 {
					// TODO: Service pending interrupts
					self.apply_instruction(&repeat_inst);
					self.sub_from_reg(reg, 1);
					//dbg!(self.get_reg_u16(reg));
					/*if self.get_reg_u16(reg) == 0 {
						break;
					}*/
					if let Some(until_zero_flag) = until_zero_flag {
						if self.get_flag(Flag::Zero) == until_zero_flag { 
							break;
						}
					}
				}
			}
			Inst::MovReg32{source_h, source_l, dest_h, dest_l} => {
				let value_h = self.get_reg_u16(source_h);
				self.set_reg_u16(dest_h, value_h);
				let value_l = self.get_reg_u16(source_l);
				self.set_reg_u16(dest_l, value_l);
			}
			Inst::Call(offset) => {
				let old_ip = self.get_reg_u16(Reg::IP);
				self.push_u16(old_ip);
				// TODO: Maybe the offset should just be i16
				self.set_reg_u16(Reg::IP, (old_ip as i32 + offset) as u16);
			}
			Inst::Ret{extra_pop} => {
				let ip = self.pop_u16();
				self.set_reg_u16(Reg::IP, ip);
				self.add_to_reg(Reg::SP, extra_pop);
			}
			Inst::CallAbsolute{ip, cs} => {
				self.push_u16(self.get_reg_u16(Reg::CS));
				self.push_u16(self.get_reg_u16(Reg::IP));
				self.set_reg_u16(Reg::CS, cs);
				self.set_reg_u16(Reg::IP, ip);
			}
			Inst::RetAbsolute => {
				let ip = self.pop_u16();
				let cs = self.pop_u16();
				self.set_reg_u16(Reg::IP, ip);
				self.set_reg_u16(Reg::CS, cs);
			}
			Inst::Jump{ref condition, offset} => {
				let do_jump = if let Some(condition) = condition {
					self.test_jump_condition(condition)
				} else {
					true
				};
				
				if do_jump {
					//println!("----- JUMP -----");
					let ip = self.get_reg_u16(Reg::IP);
					self.set_reg_u16(Reg::IP, (ip as i32 + offset) as u16);
				}
			}
			Inst::JumpAbsolute{seg, displacement, offset} => {
				let mut seg_origin = self.get_seg_origin(seg);
				if let Some(displacement) = displacement {
					seg_origin += self.calculate_displacement(displacement) as u32;
				}
				let addr = (seg_origin as i32 + offset) as u32;
				let ip = self.peek_u16(addr);
				let cs = self.peek_u16(addr + 2);
				self.set_reg_u16(Reg::IP, ip);
				self.set_reg_u16(Reg::CS, cs);
				//dbg!((cs, ip));
				/*for i in 0..16 {
					println!("{:?}: {:?}", addr + i, self.peek_u8(addr + i));
				}
				panic!();*/
			}
			Inst::JumpAndDecrementUntilZero{offset, dec_reg} => {
				self.sub_from_reg(dec_reg, 1);
				let count_value = self.get_reg_u16(dec_reg);
				if count_value != 0 {
					//println!("----- JUMP AND DEC ----- {:?}", count_value);
					
					let ip = self.get_reg_u16(Reg::IP);
					self.set_reg_u16(Reg::IP, (ip as i32 + offset) as u16);
				}
			}
			Inst::JumpZeroReg{offset, reg} => {
				let value = self.get_reg_u16(reg);
				if value == 0 {
					let ip = self.get_reg_u16(Reg::IP);
					self.set_reg_u16(Reg::IP, (ip as i32 + offset) as u16);
				}
			}
			Inst::Interrupt(interrupt_index) => {
				self.interrupt(interrupt_index);
			}
			Inst::Halt => {
				self.halted = true;
			}
		}
	}
}

// https://en.wikipedia.org/wiki/Program_Segment_Prefix
// https://toonormal.com/2018/06/07/notes-ms-dos-dev-for-intel-8086-cpus-using-a-modern-pc/
// - "DOS programs require that all programs start at the 256 byte boundary"
// https://www.daniweb.com/programming/software-development/threads/291076/whats-org-100h
// Super useful: http://www.mlsite.net/8086/
// https://en.wikibooks.org/wiki/X86_Assembly/Machine_Language_Conversion#Mod_/_Reg_/_R/M_tables
// https://www.felixcloutier.com/x86/rcl:rcr:rol:ror

fn main() {
    let mut file = std::fs::File::open("ZZT.EXE").unwrap();
    let exe_header = MzHeader::parse(&mut file).unwrap();
    //println!("{:#?}", exe_header);
    let mut machine = Machine8086::new(1024*1024*1);
    exe_header.load_into_machine(&mut machine, &mut file);
    loop {
		let inst = machine.parse_instruction();
		//println!("{:?}", inst);
		machine.apply_instruction(&inst);
	}
}