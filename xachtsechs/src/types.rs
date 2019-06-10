use crate::machine8086::Machine8086;

use num::FromPrimitive;
use num_derive::FromPrimitive;

pub trait AnyUnsignedInt8086
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
pub enum Reg {
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

pub const REG_COUNT: usize = 14;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum RegHalf {
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
	pub fn reg16(index: usize) -> Option<Reg> {
		REG16_MAPPING.get(index).map(|v| *v)
	}
	
	pub fn reg8(index: usize) -> Option<(Reg, RegHalf)> {
		REG8_MAPPING.get(index).map(|v| *v)
	}
	
	pub fn seg_reg(index: usize) -> Option<Reg> {
		SREG_MAPPING.get(index).map(|v| *v)
	}
}

pub fn treat_u8_as_i8(value: u8) -> i8 {
	unsafe { std::mem::transmute(value) }
}

pub fn treat_u16_as_i16(value: u16) -> i16 {
	unsafe { std::mem::transmute(value) }
}

pub fn treat_u32_as_i32(value: u32) -> i32 {
	unsafe { std::mem::transmute(value) }
}

pub fn treat_i8_as_u8(value: i8) -> u8 {
	unsafe { std::mem::transmute(value) }
}

pub fn treat_i16_as_u16(value: i16) -> u16 {
	unsafe { std::mem::transmute(value) }
}

pub fn treat_i32_as_u32(value: i32) -> u32 {
	unsafe { std::mem::transmute(value) }
}

pub fn split_u16_high_low(value: u16) -> (u8, u8) {
	(((value >> 8) & 0xff) as u8, (value & 0xff) as u8)
}

#[derive(Debug, Clone, PartialEq)]
pub struct FullReg {
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
pub enum OpSize {
	Size8,
	Size16,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum OpDirection {
	// This will swap the two arguments of the ModR/M byte, so that the reg bits are the second argument.
	Source,
	Destination,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum DisplacementOrigin {
	BX_SI,
	BX_DI,
	BP_SI,
	BP_DI,
	SI,
	DI,
	BP,
	BX,
}

impl DisplacementOrigin {
	pub fn default_segment(self) -> Reg {
		match self {
			DisplacementOrigin::BP_SI | DisplacementOrigin::BP_DI | DisplacementOrigin::BP => Reg::SS,
			_ => Reg::DS,
		}
	}
}

// https://www.daenotes.com/electronics/digital-electronics/8086-8088-microprocessor
#[derive(Debug, Copy, Clone, PartialEq)]
#[repr(u8)]
pub enum Flag {
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
pub fn u8_on_bits_are_odd(value: u8) -> bool {
	U8_ON_BITS_ARE_ODD_LOOKUP[value as usize]
}

#[derive(Debug, Copy, Clone, PartialEq, FromPrimitive)]
#[repr(u8)]
pub enum JumpConditionType {
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
pub struct JumpCondition {
	pub condition: JumpConditionType,
	pub negate: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, FromPrimitive)]
#[repr(u8)]
pub enum ArithmeticMode {
	Add = 0,
	Or,
	AddWithCarry,
	SubWithBorrow,
	And,
	Sub,
	Xor,
	Cmp,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ScalarMode {
	Mul,
	SignedMul,
	Div,
	SignedDiv,
}

#[derive(Debug, Copy, Clone, PartialEq, FromPrimitive)]
#[repr(u8)]
pub enum ShiftMode {
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AdjustMode {
	Addition,
	Subtraction,
}

// https://en.wikibooks.org/wiki/X86_Assembly/Machine_Language_Conversion#Mod_/_Reg_/_R/M_tables
// MOD (2 bits): Register mode
// REG (3 bits): Register
// R/M (3 bits): Register/Memory operand

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ModRmRegMode {
	Reg,
	Seg,
	Imm,
}

// This is a memory address without the segment applied to it.
#[derive(Debug, Clone, PartialEq)]
pub enum Address16 {
	Reg(Reg),
	DisplacementRel(DisplacementOrigin, u16),
	Immediate(u16),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataLocation8 {
	Reg(Reg, RegHalf),
	MemoryAbs(u32),
	Memory{seg: Reg, address16: Address16},
	Immediate(u8),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataLocation16 {
	Reg(Reg),
	MemoryAbs(u32),
	Memory{seg: Reg, address16: Address16},
	Address16(Address16),
	Immediate(u16),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataLocation {
	Size8(DataLocation8),
	Size16(DataLocation16),
}

impl DataLocation {
	pub fn with_immediate_source(self, imm: u16) -> SourceDestination {
		match self {
			DataLocation::Size8(destination) => SourceDestination::Size8(DataLocation8::Immediate(imm as u8), destination),
			DataLocation::Size16(destination) => SourceDestination::Size16(DataLocation16::Immediate(imm), destination),
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
pub enum SourceDestination {
	Size8(DataLocation8, DataLocation8),
	Size16(DataLocation16, DataLocation16),
}

impl SourceDestination {
	pub fn split(self) -> (DataLocation, DataLocation) {
		match self {
			SourceDestination::Size8(source, destination) => (DataLocation::Size8(source), DataLocation::Size8(destination)),
			SourceDestination::Size16(source, destination) => (DataLocation::Size16(source), DataLocation::Size16(destination)),
		}
	}
}

#[derive(Debug, Clone, PartialEq)]
pub enum Inst {
	NoOp,
	Push16(DataLocation16),
	Pop16(DataLocation16),
	// Popping into the flags register has some special behaviour.
	// See https://www.felixcloutier.com/x86/popf:popfd:popfq
	PopFlags,
	PushAllGeneralPurposeRegisters,
	PopAllGeneralPurposeRegisters,
	Mov(SourceDestination),
	Swap(SourceDestination),
	
	CmpAndIncrement(SourceDestination, Reg, Option<Reg>),
	/// Same as Mov, but increments the Reg by 1 or 2 for 8 and 16 bit SourceDestinations,
	/// respectively. It will do the same to the Option<Reg> if it is also set. If the Direction
	/// flag is set, it will decrement instead.
	// TODO: Maybe this should just match the SourceDestination for Addr locations and increment the
	// associated registers?
	MovAndIncrement(SourceDestination, Reg, Option<Reg>),
	MovReg32{source_h: Reg, source_l: Reg, dest_h: Reg, dest_l: Reg},
	
	Load32{seg: Reg, address16: Address16, out_reg_h: Reg, out_reg_l: Reg},
	
	Arithmetic(ArithmeticMode, SourceDestination),
	/// Use the 16-bit number `value_quot_rem` and scale it by the value in `by`,
	/// then put the quotient in the low byte of `value_quot_rem`, and the remainder in the high
	/// half of `value_quot_rem`.
	ScalarOperation8{mode: ScalarMode, value_quot_rem: Reg, by: DataLocation8},
	/// Use the 32-bit number `value_high_rem:value_low_quot` and scale it by the value in `by`,
	/// then put the quotient in value_low_quot, and the remainder in value_high_rem.
	ScalarOperation16{mode: ScalarMode, value_low_quot: Reg, value_high_rem: Reg, by: DataLocation16},
	Negate(DataLocation),
	NegateSigned(DataLocation),
	Inc(DataLocation),
	Dec(DataLocation),
	Rotate{by: DataLocation8, target: DataLocation, mode: ShiftMode},
	BitwiseCompareWithAnd(SourceDestination),
	// If you have a register with two bytes [5, 6] and you pass 10 as the base, this will set the
	// value of the register to 56, ie. 0b00111000.
	CombineBytesAsNumberWithBase(Reg, u8),
	AsciiAdjustAfter(Reg, AdjustMode),
	DecimalAdjustAfter(Reg, RegHalf, AdjustMode),
	//DecBy16(DataLocation16, u16),
	//IncBy16(DataLocation16, u16),
	SetFlag(Flag, bool),
	InvertFlag(Flag),
	RepeatNextRegTimes{reg: Reg, until_zero_flag: Option<bool>},
	Call(i32),
	// After popping the IP from the stack, it will pop extra_pop more bytes from the stack. These
	// would probably be arguments that were pushed to the stack just before calling the function in
	// the first place.
	Ret{extra_pop: u16},
	CallAbsolute{ip: u16, cs: u16},
	// From the memory address, read two bytes as the IP and CS values, then do the same thing as
	// CallAbsolute does.
	CallAbsoluteWithAddress{seg: Reg, address16: Address16},
	RetAbsolute{extra_pop: u16},
	Jump{condition: Option<JumpCondition>, offset: i32},
	JumpAbsolute{seg: Reg, address16: Address16},
	JumpAndDecrementUntilZero{offset: i32, dec_reg: Reg},
	JumpZeroReg{offset: i32, reg: Reg},
	Interrupt(u8),
	InterruptIf(u8, Flag),
	ReturnFromInterrupt,
	Halt,
	// Get the lower half of the register and sign-extend it to 16 bits and put that value back in
	// the register.
	SignExtend8To16{source: DataLocation8, destination: DataLocation16},
	SignExtend16To32{source: DataLocation16, destination_high: DataLocation16, destination_low: DataLocation16},
	// Port input into a byte will just crop the input, since the input is 2 bytes.
	PortInput{port_index: DataLocation16, destination: DataLocation},
	PortOutput{port_index: DataLocation16, source: DataLocation},
	// Override the default segment for the next instruction.
	// https://www.quora.com/Why-is-a-segment-override-prefix-used-with-an-example-in-8086-microprocessor
	//OverrideNextDefaultSegment(Reg)
}

pub trait EventHandler {
	fn handle_interrupt(&mut self, machine: &mut Machine8086, interrupt_index: u8);
	
	fn handle_port_input(&mut self, machine: &mut Machine8086, port_index: u16) -> u16;
	
	fn handle_port_output(&mut self, machine: &mut Machine8086, port_index: u16, value: u16);
}

pub enum StepResult {
	Instruction,
	Interrupt,
}
